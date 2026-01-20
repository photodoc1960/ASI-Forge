"""
Training Data Collector - Phase 1: Bootstrap from GPT-2

Captures agent state, GPT-2 decisions, and outcomes during autonomous operation.
This data is used to train HRM through imitation learning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects training data from GPT-2 decisions during autonomous operation.

    Each training example consists of:
    - State: Agent's current state (goals, curiosity, knowledge, history)
    - Action: Decision made by GPT-2
    - Result: Outcome of the action (success, knowledge gained, etc.)
    """

    def __init__(self, save_path: str = "data/hrm_training_data.json"):
        self.save_path = Path(save_path)
        self.examples: List[Dict[str, Any]] = []
        self.collection_enabled = True

        # Load existing data if available
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.examples = data.get('examples', [])
                    logger.info(f"Loaded {len(self.examples)} existing training examples")
            except Exception as e:
                logger.warning(f"Could not load existing training data: {e}")

        logger.info(f"TrainingDataCollector initialized (save_path={save_path})")

    def add_example(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Add a training example.

        Args:
            state: Agent state before decision (goals, curiosity, knowledge, etc.)
            action: Action chosen by GPT-2
            result: Outcome of executing the action
        """
        if not self.collection_enabled:
            return

        example = {
            'state': state,
            'action': action,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'example_id': len(self.examples)
        }

        self.examples.append(example)

        # Auto-save every 100 examples
        if len(self.examples) % 100 == 0:
            self.save()
            logger.info(f"Collected {len(self.examples)} training examples (auto-saved)")

    def encode_agent_state(
        self,
        agent,
        knowledge_system,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Encode current agent state into training example format.

        Args:
            agent: The autonomous agent
            knowledge_system: Knowledge augmentation system
            iteration: Current iteration number

        Returns:
            Dictionary with encoded state features
        """
        # Goals
        active_goals = agent.goal_generator.active_goals[:5]
        goal_descriptions = [g.description for g in active_goals]
        goal_types = [g.goal_type.value for g in active_goals]

        # Curiosity
        knowledge_gaps = list(agent.curiosity.knowledge_gaps)[:3]
        curiosity_descriptions = knowledge_gaps  # Already strings

        # Knowledge
        kb = knowledge_system.knowledge_base
        knowledge_count = len(kb.knowledge)

        # Get recent knowledge (last 3 items)
        recent_knowledge = []
        if knowledge_count > 0:
            all_knowledge = list(kb.knowledge.values())
            recent_items = all_knowledge[-3:] if len(all_knowledge) >= 3 else all_knowledge
            recent_knowledge = [k.content[:100] for k in recent_items]

        # Action history
        recent_actions = []
        if hasattr(agent, 'action_history'):
            recent_actions = [a.get('action', 'unknown') for a in agent.action_history[-5:]]

        # Questions asked (from curiosity)
        questions_asked = len(agent.curiosity.knowledge_gaps)

        return {
            # Goals
            'active_goals': goal_descriptions,
            'goal_types': goal_types,
            'goal_count': len(active_goals),

            # Curiosity
            'curiosity_count': len(agent.curiosity.knowledge_gaps),
            'top_curiosity': curiosity_descriptions,

            # Knowledge
            'knowledge_count': knowledge_count,
            'recent_knowledge': recent_knowledge,

            # History
            'recent_actions': recent_actions,
            'questions_asked': questions_asked,

            # Context
            'iteration': iteration
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Save collected training data to disk.

        Args:
            path: Optional custom save path (default: self.save_path)
        """
        save_path = Path(path) if path else self.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'total_examples': len(self.examples),
                'created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'examples': self.examples
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.examples)} training examples to {save_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        if not self.examples:
            return {
                'total_examples': 0,
                'action_distribution': {},
                'avg_knowledge_per_example': 0
            }

        # Action distribution
        action_counts = {}
        total_knowledge = 0

        for example in self.examples:
            action_type = example['action'].get('action', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

            # Count knowledge gained
            result = example.get('result', {})
            knowledge_gained = result.get('knowledge_gained', 0)
            total_knowledge += knowledge_gained

        return {
            'total_examples': len(self.examples),
            'action_distribution': action_counts,
            'avg_knowledge_per_example': total_knowledge / len(self.examples),
            'collection_enabled': self.collection_enabled
        }

    def clear(self) -> None:
        """Clear all collected examples (useful for starting fresh)."""
        old_count = len(self.examples)
        self.examples = []
        logger.info(f"Cleared {old_count} training examples")

    def disable_collection(self) -> None:
        """Temporarily disable data collection."""
        self.collection_enabled = False
        logger.info("Data collection disabled")

    def enable_collection(self) -> None:
        """Re-enable data collection."""
        self.collection_enabled = True
        logger.info("Data collection enabled")
