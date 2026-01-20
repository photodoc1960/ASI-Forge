"""
Neural Integration for Autonomous Agent
Connects HRM reasoning engine to agent decision-making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class AgentTokenizer:
    """
    Simple tokenizer for agent states and actions
    Maps agent concepts to token IDs
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

        # Special tokens
        self.PAD = 0
        self.START = 1
        self.END = 2
        self.SEP = 3

        # Action tokens (4-20)
        self.ACTION_TOKENS = {
            'web_search': 4,
            'ask_human': 5,
            'propose_improvement': 6,
            'idle': 7,
            'learn': 8,
            'explore': 9,
        }

        # Goal type tokens (20-40)
        self.GOAL_TOKENS = {
            'knowledge_acquisition': 20,
            'self_improvement': 21,
            'exploration': 22,
            'understanding': 23,
            'capability_improvement': 24,
        }

        # Context tokens (40-100)
        self.CONTEXT_TOKENS = {
            'goal': 40,
            'query': 41,
            'topic': 42,
            'knowledge_gap': 43,
            'motivation': 44,
            'priority': 45,
            'progress': 46,
        }

        # Create reverse mappings
        self.id_to_action = {v: k for k, v in self.ACTION_TOKENS.items()}
        self.id_to_goal = {v: k for k, v in self.GOAL_TOKENS.items()}
        self.id_to_context = {v: k for k, v in self.CONTEXT_TOKENS.items()}

        # Word token mapping (100-vocab_size)
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_word_id = 100

    def encode_text(self, text: str) -> List[int]:
        """Encode text into token IDs"""

        # Simple word-level tokenization
        words = text.lower().split()
        token_ids = []

        for word in words:
            # Check if it's a special token
            if word in self.ACTION_TOKENS:
                token_ids.append(self.ACTION_TOKENS[word])
            elif word in self.GOAL_TOKENS:
                token_ids.append(self.GOAL_TOKENS[word])
            elif word in self.CONTEXT_TOKENS:
                token_ids.append(self.CONTEXT_TOKENS[word])
            else:
                # Regular word
                if word not in self.word_to_id:
                    if self.next_word_id < self.vocab_size:
                        self.word_to_id[word] = self.next_word_id
                        self.id_to_word[self.next_word_id] = word
                        self.next_word_id += 1
                    else:
                        # Vocabulary full, use unknown token
                        token_ids.append(99)  # UNK
                        continue

                token_ids.append(self.word_to_id[word])

        return token_ids

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""

        words = []
        for token_id in token_ids:
            if token_id == self.PAD or token_id == self.START or token_id == self.END:
                continue
            elif token_id in self.id_to_action:
                words.append(self.id_to_action[token_id])
            elif token_id in self.id_to_goal:
                words.append(self.id_to_goal[token_id])
            elif token_id in self.id_to_context:
                words.append(self.id_to_context[token_id])
            elif token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append('[UNK]')

        return ' '.join(words)

    def encode_agent_state(
        self,
        current_goal: Optional[Dict[str, Any]],
        knowledge_gaps: List[str],
        recent_actions: List[str],
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Encode full agent state into token sequence

        Format:
        [START] goal <goal_type> <description> [SEP] knowledge_gap <gaps> [SEP]
        recent_actions <actions> [END]
        """

        tokens = [self.START]

        # Encode current goal
        if current_goal:
            tokens.append(self.CONTEXT_TOKENS['goal'])
            tokens.append(self.GOAL_TOKENS.get(current_goal.get('type', 'understanding'), 23))

            desc_tokens = self.encode_text(current_goal.get('description', ''))
            tokens.extend(desc_tokens[:20])  # Limit description length

            tokens.append(self.SEP)

        # Encode knowledge gaps
        if knowledge_gaps:
            tokens.append(self.CONTEXT_TOKENS['knowledge_gap'])
            for gap in knowledge_gaps[:3]:  # Max 3 gaps
                gap_tokens = self.encode_text(gap)
                tokens.extend(gap_tokens[:15])
            tokens.append(self.SEP)

        # Encode recent actions
        if recent_actions:
            for action_str in recent_actions[-3:]:  # Last 3 actions
                action_tokens = self.encode_text(action_str)
                tokens.extend(action_tokens[:10])
            tokens.append(self.SEP)

        tokens.append(self.END)

        # Pad or truncate to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.END]
        else:
            tokens = tokens + [self.PAD] * (max_length - len(tokens))

        return torch.tensor([tokens], dtype=torch.long)


class ActionDecoder:
    """
    Decodes HRM outputs into structured actions
    """

    def __init__(self, tokenizer: AgentTokenizer):
        self.tokenizer = tokenizer

    def decode_action(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Decode HRM logits into structured action

        Args:
            logits: (batch, seq_len, vocab_size) from HRM
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Structured action dictionary
        """

        # Get the last position (next token prediction)
        next_token_logits = logits[0, -1, :] / temperature

        # Top-k sampling
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Sample from top-k
        sampled_idx = torch.multinomial(top_k_probs, 1)
        predicted_token = top_k_indices[sampled_idx].item()

        # Check if it's an action token
        if predicted_token in self.tokenizer.id_to_action:
            action_type = self.tokenizer.id_to_action[predicted_token]

            # For actions that need parameters, generate them
            if action_type == 'web_search':
                query = self._generate_query_from_context(logits)
                return {
                    'action': 'web_search',
                    'query': query,
                    'confidence': top_k_probs[sampled_idx].item()
                }

            elif action_type == 'ask_human':
                question = self._generate_question_from_context(logits)
                return {
                    'action': 'ask_human',
                    'question': question,
                    'motivation': 'Neural reasoning suggests this question',
                    'confidence': top_k_probs[sampled_idx].item()
                }

            elif action_type == 'propose_improvement':
                return {
                    'action': 'propose_improvement',
                    'aspect': 'architecture',  # Could be generated
                    'motivation': 'Neural reasoning suggests improvement',
                    'confidence': top_k_probs[sampled_idx].item()
                }

            else:
                return {
                    'action': action_type,
                    'confidence': top_k_probs[sampled_idx].item()
                }

        # Fallback: decode full sequence and parse
        predicted_tokens = logits.argmax(dim=-1)[0].tolist()
        action_text = self.tokenizer.decode_tokens(predicted_tokens)

        return self._parse_action_from_text(action_text)

    def _generate_query_from_context(self, logits: torch.Tensor) -> str:
        """Generate search query from HRM hidden states"""

        # Simple approach: decode last few tokens
        predicted_tokens = logits[0, -10:, :].argmax(dim=-1).tolist()
        query_text = self.tokenizer.decode_tokens(predicted_tokens)

        # Clean up
        query_text = query_text.replace('[UNK]', '').strip()

        if not query_text:
            return "neural architecture search"  # Default

        return query_text

    def _generate_question_from_context(self, logits: torch.Tensor) -> str:
        """Generate question from HRM context"""

        predicted_tokens = logits[0, -15:, :].argmax(dim=-1).tolist()
        question_text = self.tokenizer.decode_tokens(predicted_tokens)

        question_text = question_text.replace('[UNK]', '').strip()

        if not question_text or '?' not in question_text:
            return "What should I learn next?"

        return question_text

    def _parse_action_from_text(self, text: str) -> Dict[str, Any]:
        """Parse action from decoded text"""

        # Look for action keywords
        for action_name in self.tokenizer.ACTION_TOKENS.keys():
            if action_name in text.lower():
                if action_name == 'web_search':
                    # Extract query after action
                    parts = text.lower().split(action_name)
                    query = parts[1].strip() if len(parts) > 1 else "general query"
                    return {'action': 'web_search', 'query': query}

                elif action_name == 'ask_human':
                    parts = text.lower().split(action_name)
                    question = parts[1].strip() if len(parts) > 1 else "What should I do?"
                    return {'action': 'ask_human', 'question': question, 'motivation': 'Neural decision'}

                else:
                    return {'action': action_name}

        # Default: idle
        return {'action': 'idle', 'reason': 'Could not decode action from HRM output'}


class NeuralGoalGenerator:
    """
    Generates goals using HRM reasoning
    """

    def __init__(
        self,
        reasoning_engine: nn.Module,
        tokenizer: AgentTokenizer
    ):
        self.reasoning_engine = reasoning_engine
        self.tokenizer = tokenizer

    def generate_goal_from_gaps(
        self,
        knowledge_gaps: List[str],
        interests: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Use HRM to generate a goal based on knowledge gaps

        Args:
            knowledge_gaps: List of knowledge gap descriptions
            interests: Current interest strengths

        Returns:
            Generated goal dictionary
        """

        # Encode prompt for goal generation
        prompt = f"knowledge_gap {' '.join(knowledge_gaps[:3])} goal"
        input_ids = self.tokenizer.encode_agent_state(
            current_goal=None,
            knowledge_gaps=knowledge_gaps,
            recent_actions=[],
            max_length=64
        )

        # Generate with HRM
        with torch.no_grad():
            outputs = self.reasoning_engine(input_ids)

        # Decode goal
        logits = outputs['logits']

        # Check for goal type tokens
        next_token_logits = logits[0, -1, :]
        goal_token_ids = list(self.tokenizer.GOAL_TOKENS.values())

        # Find most likely goal type
        goal_probs = F.softmax(next_token_logits[goal_token_ids], dim=-1)
        best_goal_idx = goal_probs.argmax().item()
        goal_type_id = goal_token_ids[best_goal_idx]
        goal_type = self.tokenizer.id_to_goal[goal_type_id]

        # Generate description
        description_tokens = logits[0, -20:, :].argmax(dim=-1).tolist()
        description = self.tokenizer.decode_tokens(description_tokens)

        if not description or len(description) < 5:
            # Fallback to gap-based description
            if knowledge_gaps:
                description = f"Learn about {knowledge_gaps[0].split(':')[0]}"
            else:
                description = "Explore new knowledge areas"

        return {
            'type': goal_type,
            'description': description,
            'motivation': 'Generated by neural reasoning',
            'priority': goal_probs[best_goal_idx].item(),
            'confidence': goal_probs[best_goal_idx].item()
        }
