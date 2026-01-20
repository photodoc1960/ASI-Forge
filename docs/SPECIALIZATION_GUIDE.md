# ASI-Forge User Guide

## What is ASI-Arch?

**ASI-Arch** (Autonomous Scientific Intelligence Architecture) is a groundbreaking multi-agent framework developed by researchers at GAIR-NLP that enables Large Language Models to conduct **end-to-end scientific research autonomously**.

The system operates through a continuous research loop:

```
Sample → Evolve → Evaluate → Analyze → Update → (repeat)
```

Eight specialized LLM agents work together:
- **Planner**: Designs novel architectures based on experimental evidence
- **Code Checker**: Validates implementations for correctness
- **Motivation Checker**: Ensures research directions are novel
- **Deduplication**: Prevents redundant explorations
- **Trainer**: Executes training runs
- **Debugger**: Automatically fixes errors
- **Analyzer**: Interprets experimental results
- **Summarizer**: Synthesizes insights for future experiments

In its original form, ASI-Arch was specialized for **linear attention architecture research**, where it successfully discovered **106 novel architectures** that achieve state-of-the-art performance. The system demonstrated that AI can autonomously conduct meaningful scientific research - forming hypotheses, running experiments, learning from results, and iterating toward better solutions.

For more details, see the original paper: [AlphaGo Moment for Model Architecture Discovery](https://arxiv.org/abs/2507.18074)

---

## What is ASI-Forge?

**ASI-Forge** extends ASI-Arch into a **universal meta-research framework** that can bootstrap itself into **any research domain** - not just linear attention.

### The Key Innovation

While ASI-Arch's agents and pipeline are domain-agnostic in principle, the original implementation had domain-specific elements hardcoded throughout:
- Prompts referenced "DeltaNet", "linear attention", and specific constraints
- The knowledge base contained only attention mechanism papers
- Evaluation metrics were specific to language model benchmarks
- Training infrastructure assumed PyTorch neural network training

ASI-Forge abstracts all of these into a **specialization system** that can be configured for any domain:

| Component | ASI-Arch (Original) | ASI-Forge |
|-----------|---------------------|-----------|
| Prompts | Hardcoded for linear attention | Auto-generated per domain |
| Knowledge Base | Attention mechanism papers | Auto-built via web search + user docs |
| Constraints | O(N log N) complexity, causal masking | Domain-specific, auto-generated |
| Training | Fixed PyTorch script | Configurable per domain |
| Domains | Linear attention only | Any research domain |

### Initialization Modes

ASI-Forge supports two ways to start research in a new domain:

- **Seeded Mode**: You provide an existing codebase, and ASI-Forge evolves it
- **Genesis Mode**: ASI-Forge generates an initial codebase from scratch based on your domain description

### How It Works

When you create a new specialization, ASI-Forge runs a **6-stage bootstrapping pipeline**:

1. **Domain Understanding**: LLM analyzes your description and any seed code
2. **Knowledge Acquisition**: Searches the web for relevant papers, processes your reference documents
3. **Constraint Definition**: Generates domain-appropriate validation rules
4. **Prompt Generation**: Creates all 8 agent prompts tailored to your domain
5. **Infrastructure Setup**: Configures training scripts and evaluation
6. **Validation**: Runs an end-to-end test experiment

The result is a fully configured specialization ready for autonomous research.

---

## Quick Start

### Running the Interactive CLI

```bash
cd pipeline
python -m cli
```

This opens the main menu where you can:
- View and select existing specializations
- Create new specializations
- Run the research pipeline
- Export/import specializations

### Running with a Specific Specialization

```bash
# Run pipeline with a specific specialization
python -m pipeline --spec=linear_attention

# Run in legacy mode (no specialization)
python -m pipeline --legacy

# List available specializations
python -m cli --list
```

## Creating a New Specialization

### Step 1: Launch the Creation Wizard

From the CLI main menu, select `[N] Create New Specialization`.

### Step 2: Provide Basic Information

- **Unique name**: A lowercase identifier (e.g., `drug_discovery`, `protein_folding`)
- **Display name**: Human-readable name (e.g., "Drug Discovery Research")
- **Description**: A detailed description of the research domain. This is critical - the system uses this to understand your domain and generate appropriate prompts, constraints, and infrastructure.

**Example description:**
```
Autonomous research framework for discovering novel small molecule drug candidates.
Focuses on generating molecules with desired pharmacological properties including
binding affinity, selectivity, ADMET properties, and synthetic accessibility.
The system should evolve molecular structures using SMILES representations and
evaluate them against target protein binding sites.
```

### Step 3: Choose Initialization Mode

#### Seeded Mode
Use this when you have an existing codebase you want to evolve:
- Provide the path to your seed codebase
- The system will analyze your code structure and adapt to it
- Your code becomes the starting point for evolution

#### Genesis Mode
Use this when starting from scratch:
- The system generates an initial codebase based on your domain description
- Includes best practices found through web search
- Creates a working starting point for evolution

### Step 4: Provide Reference Documents (Optional)

You can enhance the knowledge base with your own documents:

**Option 1: Folder of Documents**
- Point to a folder containing research papers, documentation, code examples
- Supported formats: `.pdf`, `.txt`, `.md`, `.json`, `.py`, `.tex`, `.rst`, `.html`, `.xml`
- The system recursively scans the folder and processes up to 50 documents

**Option 2: Individual Files**
- Add specific files one at a time
- Useful when you have a few key papers

**Option 3: Skip**
- Rely on automatic web search to build the knowledge base

### Step 5: Automatic Bootstrapping

The system runs a 6-stage pipeline to generate your specialization:

1. **Domain Understanding**: Analyzes your description and seed code
2. **Knowledge Acquisition**: Searches for relevant papers and processes your documents
3. **Constraint Definition**: Generates domain-specific validation rules
4. **Prompt Generation**: Creates all 8 specialized prompts
5. **Infrastructure Setup**: Configures training scripts and evaluation
6. **Validation**: Runs end-to-end test to verify everything works

This process may take several minutes.

## Specialization Structure

Each specialization is stored in `specializations/<name>/` with:

```
specializations/
└── your_domain/
    ├── config.json           # Main configuration
    ├── prompts/              # 8 prompt template files
    │   ├── planner.txt
    │   ├── checker.txt
    │   ├── motivation_checker.txt
    │   ├── deduplication.txt
    │   ├── debugger.txt
    │   ├── analyzer.txt
    │   ├── summarizer.txt
    │   └── model_judger.txt
    ├── knowledge/            # Domain knowledge corpus
    │   ├── doc_001_*.json
    │   └── metadata.json
    ├── infrastructure/       # Training scripts
    │   └── train.sh
    ├── seed/                 # Initial codebase
    │   └── model.py
    ├── results/              # Experiment results
    ├── debug/                # Debug logs
    └── pool/                 # Successful code versions
```

## Configuration Options

### config.json Structure

```json
{
  "id": "unique_id",
  "name": "specialization_name",
  "display_name": "Human Readable Name",
  "description": "Full domain description",
  "init_mode": "seeded|genesis",

  "architecture": {
    "base_class_name": "ModelClass",
    "artifact_type": "neural_network|molecule|device|...",
    "standard_parameters": ["param1", "param2"],
    "interface_signature": "def forward(self, x)",
    "required_decorators": ["@torch.compile"],
    "file_extension": ".py"
  },

  "evaluation": {
    "baseline_models": [...],
    "benchmarks": ["benchmark1", "benchmark2"],
    "primary_metric": "score",
    "scoring_weights": {
      "performance": 0.3,
      "innovation": 0.25,
      "complexity": 0.45
    }
  },

  "constraints": {
    "complexity_requirement": "O(N log N)",
    "strict_constraints": [...],
    "critical_constraints": [...],
    "flexible_constraints": [...],
    "preservation_rules": [...]
  },

  "infrastructure": {
    "source_file": "path/to/evolve",
    "training_script": "path/to/train.sh",
    "result_file": "path/to/loss.csv",
    "test_result_file": "path/to/benchmark.csv",
    "timeout_seconds": 7200,
    "max_debug_attempts": 3
  }
}
```

### Constraint Types

- **Strict Constraints**: Must be fixed immediately. Violations block progress.
- **Critical Constraints**: Should be fixed. Important for quality.
- **Flexible Constraints**: Nice to have. Allow for innovation.

## Managing Specializations

### Switching Between Specializations

From the CLI, simply select a specialization by number. The system saves your current work before switching.

### Exporting a Specialization

```bash
# From CLI: Select [E] Export/Import > [E] Export
# Or programmatically:
python -c "from specialization import get_manager; get_manager().export('my_spec', 'my_spec.zip')"
```

### Importing a Specialization

```bash
# From CLI: Select [E] Export/Import > [I] Import
# Or programmatically:
python -c "from specialization import get_manager; get_manager().import_specialization('my_spec.zip')"
```

### Deleting a Specialization

From the CLI, select `[D] Delete Specialization`. You must confirm by typing the specialization name.

**Warning**: This permanently deletes all experiments and data for that specialization.

## Customizing Prompts

After creation, you can manually edit the prompt files in `specializations/<name>/prompts/`. Each prompt uses placeholders that are filled at runtime:

- `{context}` - Experimental history and evidence
- `{motivation}` - Design motivation for current experiment
- `{name}` - Model/artifact name
- `{result}` - Experiment results

## Running the Pipeline

### From CLI

1. Select a specialization (or it uses the active one)
2. Choose `[R] Run Pipeline`
3. Press Ctrl+C to stop

### From Command Line

```bash
python -m pipeline --spec=linear_attention
```

### Programmatically

```python
import asyncio
from specialization import SpecializationManager
from config import Config

async def main():
    manager = SpecializationManager()
    spec = await manager.activate("my_specialization")
    Config.load_specialization(spec)

    # Now run your pipeline
    from pipeline import run_single_experiment
    await run_single_experiment()

asyncio.run(main())
```

## Validation

Before using a specialization in production, validate it:

```bash
# From CLI: Select [V] Validate Specialization
```

Validation runs a complete end-to-end test:
1. Generates a sample architecture/artifact
2. Runs training/evaluation
3. Analyzes results
4. Updates database

If validation fails, you'll see specific errors to fix.

## Troubleshooting

### "Specialization not validated" Warning

Run validation from the CLI or set `allow_unvalidated=True` when creating the manager:

```python
manager = SpecializationManager(allow_unvalidated=True)
```

### PDF Processing Not Working

Install a PDF library:

```bash
pip install PyPDF2
# or
pip install pdfplumber
```

### Knowledge Base Empty

- Ensure your reference documents are in supported formats
- Check that the folder path is correct
- Verify internet connectivity for web search

### Training Fails

1. Check `specializations/<name>/debug/error.txt` for error logs
2. Verify your training script is executable
3. Ensure all dependencies are installed

## Best Practices

1. **Write detailed descriptions**: The more context you provide, the better the generated prompts and constraints.

2. **Provide quality reference documents**: Include foundational papers and recent breakthroughs in your domain.

3. **Start with seeded mode if possible**: It's easier to evolve existing good code than generate from scratch.

4. **Run validation before production**: Catches configuration issues early.

5. **Back up before major changes**: Use the export feature to save snapshots.

6. **Review generated prompts**: The auto-generated prompts are a starting point - customize them for your specific needs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                    Specialization Manager                        │
│  • Load/save specializations                                     │
│  • Switch between domains                                        │
│  • Create new specializations                                    │
├─────────────────────────────────────────────────────────────────┤
│                    Domain Bootstrapper                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Domain    │ │  Knowledge  │ │ Constraint  │               │
│  │Understanding│ │ Acquisition │ │ Definition  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Prompt    │ │Infrastructure│ │ Validation  │               │
│  │ Generation  │ │   Setup     │ │   Agent     │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    Core Pipeline (Domain-Agnostic)              │
│              Evolve → Evaluate → Analyze → Update               │
└─────────────────────────────────────────────────────────────────┘
```

## Getting Help

- Check this guide for common issues
- Review the example `linear_attention` specialization for reference
- Examine generated config.json and prompts for understanding

## Examples

### Example: Creating a Drug Discovery Specialization

```
Name: drug_discovery
Display Name: Small Molecule Drug Discovery

Description:
Autonomous research framework for discovering novel small molecule drug
candidates targeting specific protein binding sites. The system evolves
molecular structures represented as SMILES strings, optimizing for:
- Binding affinity to target protein
- Selectivity over off-targets
- Drug-likeness (Lipinski's rules)
- Synthetic accessibility
- ADMET properties

The evaluation pipeline uses molecular docking simulations and property
prediction models. Successful candidates should have high predicted
binding affinity while maintaining favorable pharmacokinetic properties.

Mode: Genesis (generate initial molecules from scratch)

Reference Folder: /path/to/drug_discovery_papers/
```

### Example: Creating a Materials Science Specialization

```
Name: battery_materials
Display Name: Battery Electrode Materials

Description:
Research framework for discovering novel battery electrode materials with
improved energy density and cycle stability. Evolves crystal structures
and compositions for lithium-ion and solid-state batteries. Evaluates
candidates using DFT calculations for voltage, capacity, and stability.

Mode: Seeded
Seed Path: /path/to/existing/materials_code/

Reference Folder: /path/to/materials_papers/
```
