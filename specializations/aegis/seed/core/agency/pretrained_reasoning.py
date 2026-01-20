"""
Pretrained LLM Integration for Agent Reasoning

Uses a small pretrained language model from HuggingFace as the reasoning engine.
This provides immediate language understanding and reasoning capabilities.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Any, Optional
import logging
import warnings

logger = logging.getLogger(__name__)


class PretrainedReasoningEngine:
    """
    Wrapper around a pretrained LLM for agent reasoning.

    Converts agent state -> text prompt -> LLM inference -> structured action
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        """
        Initialize pretrained reasoning engine

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading pretrained model: {model_name}")

        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load config and ensure loss_type is not set to None
            config = AutoConfig.from_pretrained(model_name)
            # Remove loss_type if it's None (transformers will use the correct default)
            if hasattr(config, 'loss_type') and config.loss_type is None:
                delattr(config, 'loss_type')

            # Suppress the loss_type warning from transformers
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*loss_type.*')

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=True,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )

            if device == "cpu":
                self.model = self.model.to(device)

            self.model.eval()

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✓ Model loaded successfully on {device}")
            logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def format_agent_prompt(
        self,
        current_goal: Optional[Dict[str, str]],
        knowledge_gaps: List[str],
        recent_actions: List[str],
        interests: Dict[str, float]
    ) -> str:
        """
        Format agent state as a natural language prompt

        Returns a prompt like:
        "You are an autonomous AI agent. Your current state:

        Goal: Understand my architecture (type: understanding)

        Knowledge Gaps:
        - attention mechanisms
        - learning efficiency

        Recent Actions: [idle, idle, web_search]

        Top Interests:
        - neural_architecture (0.80)
        - reasoning_strategies (0.65)

        Based on this state, what action should you take?
        Available actions: web_search, ask_human, propose_improvement, idle, learn, explore

        Action:"
        """

        prompt_parts = ["You are an autonomous AI agent making decisions. Current state:\n"]

        # Current goal
        if current_goal:
            prompt_parts.append(f"\nGoal: {current_goal.get('description', 'None')}")
            prompt_parts.append(f"Type: {current_goal.get('type', 'unknown')}\n")
        else:
            prompt_parts.append("\nGoal: None (need to generate new goal)\n")

        # Knowledge gaps
        if knowledge_gaps:
            prompt_parts.append("\nKnowledge Gaps:")
            for gap in knowledge_gaps[:5]:  # Limit to 5
                prompt_parts.append(f"- {gap}")
            prompt_parts.append("")

        # Recent actions
        if recent_actions:
            actions_str = ", ".join(recent_actions[-5:])  # Last 5
            prompt_parts.append(f"Recent Actions: [{actions_str}]\n")

        # Top interests
        if interests:
            top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_interests:
                prompt_parts.append("Top Interests:")
                for topic, strength in top_interests:
                    prompt_parts.append(f"- {topic} ({strength:.2f})")
                prompt_parts.append("")

        # Action prompt - SIMPLIFIED and CLEAR for GPT-2
        # Give GPT-2 a clear template to complete
        prompt_parts.append("\nDecide action and query:")
        prompt_parts.append("ACTION: web_search")
        prompt_parts.append("QUERY:")  # GPT-2 will complete this

        return "\n".join(prompt_parts)

    def generate_action(
        self,
        current_goal: Optional[Dict[str, str]],
        knowledge_gaps: List[str],
        recent_actions: List[str],
        interests: Dict[str, float],
        temperature: float = 0.7,
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Generate action based on agent state

        Returns:
            Dict with action, parameters, and confidence
        """

        # Format prompt
        prompt = self.format_agent_prompt(
            current_goal=current_goal,
            knowledge_gaps=knowledge_gaps,
            recent_actions=recent_actions,
            interests=interests
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse action from generated text
        action = self._parse_action_from_text(generated_text, current_goal)

        # Calculate confidence from generation scores
        if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
            # Average probability of generated tokens
            probs = [torch.softmax(score[0], dim=-1).max().item() for score in outputs.scores[:3]]
            confidence = sum(probs) / len(probs) if probs else 0.5
        else:
            confidence = 0.5

        action['confidence'] = confidence
        action['raw_generation'] = generated_text

        return action

    def _parse_action_from_text(self, text: str, current_goal: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """
        Parse action from LLM-generated text

        Examples:
        - "web_search(attention mechanisms)" -> {action: web_search, query: ...}
        - "ask_human" -> {action: ask_human, ...}
        - "idle" -> {action: idle}
        """

        # DEBUG: Log what GPT-2 actually generated
        logger.debug(f"GPT-2 generated text: '{text}'")

        text_lower = text.lower().strip()

        # Extract first line or sentence
        first_line = text.split('\n')[0].strip()

        # BIAS TOWARDS WEB_SEARCH: If we have a goal, default to web search
        # This ensures training data collection happens
        has_goal = current_goal is not None and current_goal.get('description')

        # Try to identify action
        if 'web_search' in text_lower or 'search' in text_lower or (has_goal and 'idle' not in text_lower):
            # Extract query
            query = self._extract_query(text, current_goal)
            return {
                'action': 'web_search',
                'query': query,
                'reasoning': first_line
            }

        elif 'ask_human' in text_lower or 'ask' in text_lower or 'question' in text_lower:
            question = self._extract_question(text, current_goal)
            return {
                'action': 'ask_human',
                'question': question,
                'reasoning': first_line
            }

        elif 'propose' in text_lower or 'improve' in text_lower or 'suggestion' in text_lower:
            # Determine what aspect to improve (context-aware)
            aspect = 'architecture'  # default

            # New proposal types
            if 'ensemble' in text_lower or 'combine' in text_lower or 'fusion' in text_lower:
                aspect = 'ensemble'
            elif 'verify' in text_lower or 'safety' in text_lower or 'check' in text_lower or 'auxiliary' in text_lower:
                aspect = 'auxiliary'
            # Original types
            elif 'attention' in text_lower or 'layer' in text_lower:
                aspect = 'architecture'
            elif 'learn' in text_lower or 'training' in text_lower or 'optimizer' in text_lower:
                aspect = 'learning'
            elif 'reasoning' in text_lower or 'thinking' in text_lower:
                aspect = 'reasoning'

            return {
                'action': 'propose_improvement',
                'aspect': aspect,
                'proposal': first_line,
                'reasoning': text
            }

        elif 'learn' in text_lower and 'web_search' not in text_lower:
            return {
                'action': 'learn',
                'focus': current_goal.get('description', 'general') if current_goal else 'general',
                'reasoning': first_line
            }

        elif 'explore' in text_lower:
            return {
                'action': 'explore',
                'domain': current_goal.get('description', 'general') if current_goal else 'general',
                'reasoning': first_line
            }

        else:
            # Default to idle
            return {
                'action': 'idle',
                'reasoning': f"No clear action identified from: {first_line}"
            }

    def _extract_query(self, text: str, current_goal: Optional[Dict[str, str]]) -> str:
        """Extract search query from generated text"""

        # DEBUG: Log extraction attempts
        logger.debug(f"Extracting query from: '{text}'")

        # First, check for the new structured format "QUERY: <text>"
        if 'QUERY:' in text or 'query:' in text.lower():
            # Find text after QUERY:
            for line in text.split('\n'):
                if 'QUERY:' in line or 'query:' in line.lower():
                    # Extract everything after "QUERY:"
                    if ':' in line:
                        query = line.split(':', 1)[1].strip().strip('"').strip("'")
                        if query and len(query) >= 5 and not query.replace('.', '').replace('-', '').isdigit():
                            logger.debug(f"Using query from QUERY: format: '{query}'")
                            return query

        # Look for parentheses
        if '(' in text and ')' in text:
            try:
                start = text.index('(') + 1
                end = text.index(')', start)
                query = text[start:end].strip().strip('"').strip("'")
                logger.debug(f"Found in parentheses: '{query}' (len={len(query)})")
                # Improved validation: must be 5+ chars and not purely numeric
                if query and len(query) >= 5 and not query.replace('.', '').replace('-', '').isdigit():
                    logger.debug(f"Using query from parentheses: '{query}'")
                    return query
                elif query:
                    logger.debug(f"Rejected query '{query}': too short or numeric")
            except ValueError:
                # ')' not found after '(' - skip this extraction method
                logger.debug("Found '(' but no matching ')' after it")

        # Look for quotes
        if '"' in text:
            parts = text.split('"')
            if len(parts) >= 2 and len(parts[1].strip()) > 3:
                logger.debug(f"Using query from quotes: '{parts[1].strip()}'")
                return parts[1].strip()

        # ALWAYS use goal description when available - this ensures good queries!
        if current_goal and current_goal.get('description'):
            goal_desc = current_goal['description']
            # Make it a searchable query
            if not any(word in goal_desc.lower() for word in ['how', 'what', 'why', 'research', 'paper']):
                query = f"research on {goal_desc}"
                logger.debug(f"Using goal-based query: '{query}'")
                return query
            logger.debug(f"Using goal description as query: '{goal_desc}'")
            return goal_desc

        logger.debug("Using fallback query: 'neural network architectures research'")
        return "neural network architectures research"

    def _extract_question(self, text: str, current_goal: Optional[Dict[str, str]]) -> str:
        """Extract question from generated text"""

        # Look for question marks
        sentences = text.split('.')
        for sentence in sentences:
            if '?' in sentence:
                return sentence.strip()

        # Look for parentheses
        if '(' in text and ')' in text:
            start = text.index('(') + 1
            end = text.index(')', start)
            question = text[start:end].strip()
            if question:
                return question

        # Fallback: generate from goal
        if current_goal and current_goal.get('description'):
            return f"Can you help me understand {current_goal['description']}?"

        return "Can you provide guidance on what I should focus on?"

    def __call__(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make compatible with HRM interface (for drop-in replacement)

        Returns logits in same format as HRM
        """
        with torch.no_grad():
            outputs = self.model(input_ids)

        return {
            'logits': outputs.logits,
            'ponder_cost': torch.tensor(0.0)  # No adaptive computation
        }
