from dataclasses import dataclass, asdict
from typing import Dict, Optional

from utils.agent_logger import log_agent_run
from .model import summarizer
from .prompt import Summary_input


@dataclass
class DataElement:
    """Data element model for experimental results."""
    time: str
    name: str
    result: Dict[str, str]
    program: str
    motivation: str
    analysis: str
    cognition: str
    log: str
    parent: Optional[int] = None
    index: Optional[int] = None
    summary: Optional[str] = None
    motivation_embedding: Optional[list] = None
    score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert DataElement instance to dictionary."""
        return asdict(self)
    
    async def get_context(self) -> str:
        """Generate enhanced context with structured experimental evidence presentation."""
        summary = await log_agent_run(
            "summarizer",
            summarizer,
            Summary_input(self.motivation, self.analysis, self.cognition)
        )
        summary_result = summary.final_output.experience

        # Safely get result fields
        result = self.result if isinstance(self.result, dict) else {}
        train_result = result.get("train", "N/A")
        test_result = result.get("test", "N/A")

        return f"""## EXPERIMENTAL EVIDENCE PORTFOLIO

### Experiment: {self.name}
**Architecture Identifier**: {self.name}

#### Performance Metrics Summary
**Training Progression**: {train_result}
**Evaluation Results**: {test_result}

#### Implementation Analysis
```python
{self.program}
```

#### Synthesized Experimental Insights
{summary_result}

---"""

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataElement':
        """Create DataElement instance from dictionary."""
        return cls(
            time=data.get('time', ''),
            name=data.get('name', ''),
            result=data.get('result', {}),
            program=data.get('program', ''),
            motivation=data.get('motivation', ''),
            analysis=data.get('analysis', ''),
            cognition=data.get('cognition', ''),
            log=data.get('log', ''),
            parent=data.get('parent', None),
            index=data.get('index', None),
            summary=data.get('summary', None),
            motivation_embedding=data.get('motivation_embedding', None),
            score=data.get('score', None)
        )