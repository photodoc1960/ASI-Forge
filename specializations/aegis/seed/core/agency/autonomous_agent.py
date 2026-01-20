"""
Autonomous Goal-Driven Agent System

This module implements a truly autonomous agent that:
- Sets its own goals based on curiosity and intrinsic motivation
- Asks questions to learn about the world
- Actively seeks knowledge through web searches
- Proposes self-improvements
- Operates continuously with human oversight
"""

import torch
import random
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of goals the agent can have"""
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    CAPABILITY_IMPROVEMENT = "capability_improvement"
    PROBLEM_SOLVING = "problem_solving"
    EXPLORATION = "exploration"
    SELF_IMPROVEMENT = "self_improvement"
    UNDERSTANDING = "understanding"


@dataclass
class Goal:
    """A goal that the agent is pursuing"""
    goal_id: str
    goal_type: GoalType
    description: str
    motivation: str  # Why the agent wants this
    created_at: datetime
    priority: float  # 0.0 to 1.0
    progress: float = 0.0  # 0.0 to 1.0
    completed: bool = False
    sub_goals: List['Goal'] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    knowledge_gained: List[str] = field(default_factory=list)


@dataclass
class Question:
    """A question the agent wants answered"""
    question: str
    motivation: str
    topic: str
    related_goal: Optional[str] = None  # Goal ID
    timestamp: datetime = field(default_factory=datetime.now)
    answered: bool = False
    answer: Optional[str] = None


@dataclass
class Interest:
    """An area of interest to the agent"""
    topic: str
    strength: float  # 0.0 to 1.0
    why_interesting: str
    first_encountered: datetime
    last_explored: datetime
    knowledge_items: List[str] = field(default_factory=list)


class CuriosityEngine:
    """
    Drives agent curiosity and generates questions
    Based on information gap theory - curiosity arises from knowledge gaps
    """

    def __init__(self):
        self.knowledge_gaps: Set[str] = set()
        self.interests: Dict[str, Interest] = {}
        self.questions_asked: List[Question] = []
        self.surprise_threshold = 0.7  # How surprising something must be to trigger curiosity

    def register_knowledge_gap(self, topic: str, gap_description: str):
        """Register a gap in knowledge that triggers curiosity"""

        gap_key = f"{topic}:{gap_description}"
        self.knowledge_gaps.add(gap_key)

        # Increase interest in this topic
        if topic not in self.interests:
            self.interests[topic] = Interest(
                topic=topic,
                strength=0.5,
                why_interesting=f"Knowledge gap detected: {gap_description}",
                first_encountered=datetime.now(),
                last_explored=datetime.now()
            )
        else:
            # Increase strength (with saturation)
            self.interests[topic].strength = min(
                1.0,
                self.interests[topic].strength + 0.1
            )

        logger.info(f"Curiosity triggered: {gap_description} (topic: {topic})")

    def generate_question(self, context: Optional[str] = None) -> Optional[Question]:
        """Generate a question based on current knowledge gaps and interests"""

        if not self.knowledge_gaps and not self.interests:
            return None

        # Choose a knowledge gap to ask about
        if self.knowledge_gaps:
            gap = random.choice(list(self.knowledge_gaps))
            topic, gap_desc = gap.split(':', 1)

            # Generate question
            questions_templates = [
                f"What is {gap_desc}?",
                f"How does {gap_desc} work?",
                f"Why is {gap_desc} important?",
                f"Can you explain {gap_desc}?",
                f"What are the key concepts in {gap_desc}?",
            ]

            question_text = random.choice(questions_templates)

            question = Question(
                question=question_text,
                motivation=f"I want to understand {topic} better",
                topic=topic
            )

            self.questions_asked.append(question)
            return question

        # Or ask about an area of interest
        if self.interests:
            topic = max(self.interests.keys(), key=lambda t: self.interests[t].strength)
            interest = self.interests[topic]

            question_text = f"What are the latest developments in {topic}?"

            question = Question(
                question=question_text,
                motivation=f"High interest in {topic}: {interest.why_interesting}",
                topic=topic
            )

            self.questions_asked.append(question)
            return question

        return None

    def process_surprise(self, observation: str, expected: str, surprise_level: float):
        """Process surprising information that doesn't match expectations"""

        if surprise_level > self.surprise_threshold:
            logger.info(f"Surprise detected (level {surprise_level:.2f}): {observation}")

            # This creates curiosity about why the surprise occurred
            self.register_knowledge_gap(
                topic="unexpected_phenomena",
                gap_description=f"Why did '{observation}' occur when I expected '{expected}'?"
            )

    def update_interest(self, topic: str, delta: float):
        """Update interest level in a topic"""

        if topic in self.interests:
            self.interests[topic].strength = max(
                0.0,
                min(1.0, self.interests[topic].strength + delta)
            )


class GoalGenerator:
    """Generates and manages agent goals"""

    def __init__(self):
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.goal_counter = 0

    def generate_goal(
        self,
        goal_type: GoalType,
        trigger: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Goal:
        """Generate a new goal"""

        self.goal_counter += 1

        # Generate goal description based on type
        if goal_type == GoalType.KNOWLEDGE_ACQUISITION:
            description = f"Learn about {context.get('topic', 'new domain')}"
            motivation = f"Expand knowledge base: {trigger}"

        elif goal_type == GoalType.CAPABILITY_IMPROVEMENT:
            description = f"Improve ability to {context.get('capability', 'reason')}"
            motivation = f"Enhance capabilities: {trigger}"

        elif goal_type == GoalType.SELF_IMPROVEMENT:
            description = f"Evolve architecture to better {context.get('aspect', 'perform tasks')}"
            motivation = f"Self-optimization: {trigger}"

        elif goal_type == GoalType.EXPLORATION:
            description = f"Explore {context.get('domain', 'unknown territory')}"
            motivation = f"Discover new information: {trigger}"

        elif goal_type == GoalType.UNDERSTANDING:
            description = f"Deeply understand {context.get('concept', 'core principles')}"
            motivation = f"Build coherent world model: {trigger}"

        else:
            description = "Generic goal"
            motivation = trigger

        goal = Goal(
            goal_id=f"goal_{self.goal_counter}",
            goal_type=goal_type,
            description=description,
            motivation=motivation,
            created_at=datetime.now(),
            priority=context.get('priority', 0.5) if context else 0.5
        )

        self.active_goals.append(goal)
        logger.info(f"New goal generated: {description}")

        return goal

    def select_next_goal(self) -> Optional[Goal]:
        """Select the next goal to pursue based on priority"""

        if not self.active_goals:
            return None

        # Sort by priority (and recency as tiebreaker)
        sorted_goals = sorted(
            [g for g in self.active_goals if not g.completed],
            key=lambda g: (g.priority, g.created_at),
            reverse=True
        )

        return sorted_goals[0] if sorted_goals else None

    def update_goal_progress(self, goal_id: str, progress: float, action: str):
        """Update progress on a goal"""

        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.progress = min(1.0, progress)
                goal.actions_taken.append(action)

                if goal.progress >= 1.0:
                    goal.completed = True
                    self.completed_goals.append(goal)
                    logger.info(f"Goal completed: {goal.description}")

                break


class IntrinsicMotivationSystem:
    """
    Provides intrinsic motivation for the agent
    Based on theories of curiosity, competence, and autonomy
    """

    def __init__(self):
        self.curiosity_drive = 0.8  # How curious the agent is
        self.competence_drive = 0.7  # Desire to improve skills
        self.autonomy_drive = 0.9  # Desire to make own decisions
        self.exploration_drive = 0.6  # Desire to explore

        # Track what motivates the agent
        self.motivation_history: List[Dict[str, Any]] = []

    def compute_motivation(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Compute intrinsic motivation for an action

        Returns:
            Motivation score 0.0 to 1.0
        """

        motivation = 0.0

        # Curiosity: Novel or information-gathering actions
        if context.get('novel', False) or 'learn' in action.lower():
            motivation += self.curiosity_drive * 0.4

        # Competence: Actions that improve capabilities
        if 'improve' in action.lower() or 'practice' in action.lower():
            motivation += self.competence_drive * 0.3

        # Autonomy: Self-directed actions
        if context.get('self_directed', True):
            motivation += self.autonomy_drive * 0.2

        # Exploration: New domains or methods
        if context.get('exploratory', False):
            motivation += self.exploration_drive * 0.1

        # Record
        self.motivation_history.append({
            'action': action,
            'motivation': motivation,
            'timestamp': datetime.now()
        })

        return min(1.0, motivation)

    def get_current_drives(self) -> Dict[str, float]:
        """Get current state of all drives"""

        return {
            'curiosity': self.curiosity_drive,
            'competence': self.competence_drive,
            'autonomy': self.autonomy_drive,
            'exploration': self.exploration_drive
        }


class AutonomousAgent:
    """
    Main autonomous agent that operates independently

    Key behaviors:
    - Generates own goals based on curiosity
    - Asks questions to learn
    - Searches for information
    - Proposes self-improvements
    - Makes decisions autonomously (with human oversight)
    """

    def __init__(self, name: str = "AEGIS Agent", reasoning_engine: Optional[Any] = None):
        self.name = name

        # Core systems
        self.curiosity = CuriosityEngine()
        self.goal_generator = GoalGenerator()
        self.motivation = IntrinsicMotivationSystem()

        # Neural reasoning (NEW!)
        self.reasoning_engine = reasoning_engine
        self.use_neural_reasoning = reasoning_engine is not None

        # Check if this is a pretrained LLM or custom HRM
        self.use_pretrained_llm = False
        if self.use_neural_reasoning:
            from core.agency.pretrained_reasoning import PretrainedReasoningEngine
            self.use_pretrained_llm = isinstance(reasoning_engine, PretrainedReasoningEngine)

        if self.use_neural_reasoning:
            if self.use_pretrained_llm:
                # Pretrained LLM - uses text-based reasoning
                logger.info(f"✓ Pretrained LLM reasoning enabled: {reasoning_engine.model_name}")
                self.tokenizer = None
                self.action_decoder = None
                self.neural_goal_generator = None
            else:
                # Custom HRM - uses token-based reasoning
                from core.agency.neural_integration import AgentTokenizer, ActionDecoder, NeuralGoalGenerator

                self.tokenizer = AgentTokenizer(vocab_size=getattr(reasoning_engine, 'vocab_size', 1000))
                self.action_decoder = ActionDecoder(self.tokenizer)
                self.neural_goal_generator = NeuralGoalGenerator(reasoning_engine, self.tokenizer)

                logger.info(f"✓ Neural reasoning enabled with {type(reasoning_engine).__name__}")
        else:
            self.tokenizer = None
            self.action_decoder = None
            self.neural_goal_generator = None
            logger.info("⚠ Running in rule-based mode (no reasoning engine provided)")

        # State
        self.is_active = False
        self.current_goal: Optional[Goal] = None
        self.pending_questions: List[Question] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.recent_actions: List[str] = []  # Track for neural context

        # Callbacks for actions (to be set by parent system)
        self.ask_human_callback: Optional[callable] = None
        self.web_search_callback: Optional[callable] = None
        self.propose_improvement_callback: Optional[callable] = None

        logger.info(f"Autonomous agent '{name}' initialized")

    def set_callbacks(
        self,
        ask_human: callable,
        web_search: callable,
        propose_improvement: callable
    ):
        """Set callbacks for agent actions"""

        self.ask_human_callback = ask_human
        self.web_search_callback = web_search
        self.propose_improvement_callback = propose_improvement

    def think(self) -> Dict[str, Any]:
        """
        Main thinking loop - decides what to do next

        Returns:
            Action to take
        """

        # Check if we have a current goal
        if self.current_goal is None or self.current_goal.completed:
            # Select or generate a new goal
            self.current_goal = self.goal_generator.select_next_goal()

            if self.current_goal is None:
                # No goals - generate one based on curiosity
                if self.use_neural_reasoning:
                    self._generate_neural_goal()
                else:
                    self._generate_curiosity_driven_goal()

                self.current_goal = self.goal_generator.select_next_goal()

        if self.current_goal is None:
            return {'action': 'idle', 'reason': 'No goals or curiosity triggers'}

        # Decide on action to pursue current goal
        if self.use_neural_reasoning:
            action = self._neural_decide_action(self.current_goal)
        else:
            action = self._decide_action_for_goal(self.current_goal)

        # Track action for neural context
        if action['action'] != 'idle':
            action_str = f"{action['action']}"
            if len(self.recent_actions) > 10:
                self.recent_actions.pop(0)
            self.recent_actions.append(action_str)

        return action

    def _generate_neural_goal(self):
        """Generate a goal using HRM neural reasoning"""
        if not self.use_neural_reasoning:
            return self._generate_curiosity_driven_goal()

        logger.info("🧠 Using neural reasoning to generate goal")

        # Use neural goal generator
        goal_dict = self.neural_goal_generator.generate_goal_from_gaps(
            knowledge_gaps=list(self.curiosity.knowledge_gaps),
            interests={k: v.strength for k, v in self.curiosity.interests.items()}
        )

        if goal_dict:
            # Convert to Goal object and add to generator
            goal_type_map = {
                'knowledge_acquisition': GoalType.KNOWLEDGE_ACQUISITION,
                'self_improvement': GoalType.SELF_IMPROVEMENT,
                'exploration': GoalType.EXPLORATION,
                'understanding': GoalType.UNDERSTANDING,
                'capability_improvement': GoalType.CAPABILITY_IMPROVEMENT
            }

            goal_type = goal_type_map.get(goal_dict['type'], GoalType.UNDERSTANDING)

            self.goal_generator.generate_goal(
                goal_type=goal_type,
                trigger=goal_dict['motivation'],
                context={
                    'description': goal_dict['description'],
                    'priority': goal_dict.get('priority', 0.5),
                    'neural_generated': True,
                    'confidence': goal_dict.get('confidence', 0.5)
                }
            )

            logger.info(f"✓ Generated neural goal: {goal_dict['type']} - {goal_dict['description']}")
        else:
            logger.warning("Neural goal generation failed, falling back to rule-based")
            self._generate_curiosity_driven_goal()

    def _neural_decide_action(self, goal: Goal) -> Dict[str, Any]:
        """Use neural reasoning (LLM or HRM) to decide action for goal"""
        if not self.use_neural_reasoning:
            return self._decide_action_for_goal(goal)

        logger.info(f"🧠 Using neural reasoning to decide action for goal: {goal.description}")

        # Use pretrained LLM for text-based reasoning
        if self.use_pretrained_llm:
            current_goal_dict = {
                'type': goal.goal_type.value if hasattr(goal.goal_type, 'value') else str(goal.goal_type),
                'description': goal.description
            }

            action = self.reasoning_engine.generate_action(
                current_goal=current_goal_dict,
                knowledge_gaps=list(self.curiosity.knowledge_gaps)[:5],
                recent_actions=self.recent_actions[-10:],
                interests={k: v.strength for k, v in self.curiosity.interests.items()}
            )

            # Add goal context
            action['goal_id'] = goal.goal_id
            action['motivation'] = goal.motivation
            action['neural_decision'] = True

            logger.info(f"✓ LLM action: {action['action']} (confidence: {action.get('confidence', 0):.2f})")
            logger.info(f"  Reasoning: {action.get('reasoning', 'N/A')}")

            return action

        # Use custom HRM for token-based reasoning
        else:
            # Encode current agent state
            current_goal_dict = {
                'type': goal.goal_type.value if hasattr(goal.goal_type, 'value') else str(goal.goal_type),
                'description': goal.description
            }

            input_ids = self.tokenizer.encode_agent_state(
                current_goal=current_goal_dict,
                knowledge_gaps=list(self.curiosity.knowledge_gaps)[:5],  # Limit to 5 most recent
                recent_actions=self.recent_actions[-10:],  # Last 10 actions
                max_length=128
            )

            # Get HRM reasoning
            import torch
            with torch.no_grad():
                outputs = self.reasoning_engine(input_ids)

            # Decode to action
            action = self.action_decoder.decode_action(
                logits=outputs['logits'],
                temperature=0.8,  # Some randomness for exploration
                top_k=5
            )

            # Add goal context
            action['goal_id'] = goal.goal_id
            action['motivation'] = goal.motivation
            action['neural_decision'] = True
            action['ponder_cost'] = outputs.get('ponder_cost', 0).item() if hasattr(outputs.get('ponder_cost', 0), 'item') else 0

            logger.info(f"✓ Neural action selected: {action['action']} (confidence: {action.get('confidence', 0):.2f})")

            return action

    def _generate_curiosity_driven_goal(self):
        """Generate a goal based on current curiosity"""

        # Check knowledge gaps
        if self.curiosity.knowledge_gaps:
            # Pick a random gap to explore
            gap = random.choice(list(self.curiosity.knowledge_gaps))
            topic, _ = gap.split(':', 1)

            self.goal_generator.generate_goal(
                goal_type=GoalType.KNOWLEDGE_ACQUISITION,
                trigger="Curiosity about knowledge gap",
                context={'topic': topic, 'priority': 0.7}
            )

        # Or explore interests
        elif self.curiosity.interests:
            topic = max(
                self.curiosity.interests.keys(),
                key=lambda t: self.curiosity.interests[t].strength
            )

            self.goal_generator.generate_goal(
                goal_type=GoalType.EXPLORATION,
                trigger=f"High interest in {topic}",
                context={'domain': topic, 'priority': 0.6}
            )

        # Or generate self-improvement goal
        else:
            self.goal_generator.generate_goal(
                goal_type=GoalType.SELF_IMPROVEMENT,
                trigger="Continuous improvement drive",
                context={'aspect': 'general capabilities', 'priority': 0.5}
            )

    def _decide_action_for_goal(self, goal: Goal) -> Dict[str, Any]:
        """Decide what action to take to advance a goal"""

        if goal.goal_type == GoalType.KNOWLEDGE_ACQUISITION:
            # Generate a question
            question = self.curiosity.generate_question(context=goal.description)

            if question:
                self.pending_questions.append(question)

                # Decide who to ask
                if random.random() < 0.7:  # Prefer web search
                    return {
                        'action': 'web_search',
                        'query': question.question,
                        'goal_id': goal.goal_id,
                        'motivation': question.motivation
                    }
                else:
                    return {
                        'action': 'ask_human',
                        'question': question.question,
                        'goal_id': goal.goal_id,
                        'motivation': question.motivation
                    }

        elif goal.goal_type == GoalType.SELF_IMPROVEMENT:
            return {
                'action': 'propose_improvement',
                'goal_id': goal.goal_id,
                'aspect': 'architecture',
                'motivation': goal.motivation
            }

        elif goal.goal_type == GoalType.EXPLORATION:
            return {
                'action': 'web_search',
                'query': f"latest research on {goal.description}",
                'goal_id': goal.goal_id,
                'motivation': goal.motivation
            }

        return {'action': 'idle', 'reason': 'No action determined'}

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action"""

        action_type = action['action']

        if action_type == 'web_search' and self.web_search_callback:
            result = self.web_search_callback(action['query'])
            self._process_search_result(action['goal_id'], result)
            return {'status': 'completed', 'result': result}

        elif action_type == 'ask_human' and self.ask_human_callback:
            result = self.ask_human_callback(action['question'], action['motivation'])
            self._process_human_answer(action['goal_id'], result)
            return {'status': 'completed', 'result': result}

        elif action_type == 'propose_improvement' and self.propose_improvement_callback:
            result = self.propose_improvement_callback(action['aspect'], action['motivation'])
            return {'status': 'proposed', 'result': result}

        else:
            return {'status': 'failed', 'reason': 'No callback for action type'}

    def _process_search_result(self, goal_id: str, result: str):
        """Process web search result"""

        # Add to knowledge base
        self.knowledge_base[f"search_{datetime.now().isoformat()}"] = result

        # Update goal progress
        self.goal_generator.update_goal_progress(
            goal_id,
            progress=0.3,
            action=f"Completed web search"
        )

        # Check for surprises or new knowledge gaps
        # (In production, use NLP to extract concepts)
        logger.info(f"Processed search result for goal {goal_id}")

    def _process_human_answer(self, goal_id: str, answer: str):
        """Process answer from human"""

        # Add to knowledge base
        self.knowledge_base[f"human_answer_{datetime.now().isoformat()}"] = answer

        # Update goal progress
        self.goal_generator.update_goal_progress(
            goal_id,
            progress=0.5,
            action=f"Received human answer"
        )

        logger.info(f"Processed human answer for goal {goal_id}")

    def get_agent_state(self) -> Dict[str, Any]:
        """Get current state of the agent"""

        return {
            'name': self.name,
            'is_active': self.is_active,
            'current_goal': {
                'description': self.current_goal.description,
                'progress': self.current_goal.progress,
                'type': self.current_goal.goal_type.value
            } if self.current_goal else None,
            'active_goals': len(self.goal_generator.active_goals),
            'completed_goals': len(self.goal_generator.completed_goals),
            'knowledge_items': len(self.knowledge_base),
            'interests': {
                topic: interest.strength
                for topic, interest in self.curiosity.interests.items()
            },
            'drives': self.motivation.get_current_drives(),
            'pending_questions': len(self.pending_questions)
        }
