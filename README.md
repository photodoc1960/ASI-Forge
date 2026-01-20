<p align="center">
 <h1 align="center">ASI-Forge</h1>
 <h3 align="center">Universal Autonomous Scientific Research Framework</h3>
</p>
<p align="center">
 <a href="https://github.com/photodoc1960/ASI-Forge/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
 <a href="https://arxiv.org/pdf/2507.18074"><img alt="Based on" src="https://img.shields.io/badge/based%20on-ASI--Arch-green"></a>
</p>

## Overview

**ASI-Forge** is a universal meta-research framework that enables LLMs to conduct autonomous scientific research in **any domain**. It extends [ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch) by Liu et al., adding a specialization system that bootstraps the research pipeline into new domains automatically.

### What ASI-Forge Does

- **Bootstraps into any research domain** - Provide a description and optional seed code; the system generates domain-specific prompts, constraints, and evaluation infrastructure
- **Runs autonomous research loops** - Eight specialized LLM agents collaborate to hypothesize, implement, evaluate, and learn from experiments
- **Maintains isolated experiment histories** - Each specialization tracks its own experiments, allowing parallel research across domains

### Initialization Modes

- **Seeded Mode**: Provide an existing codebase to evolve
- **Genesis Mode**: Generate initial code from scratch based on domain description

## Quick Start

```bash
# Clone and install
git clone https://github.com/photodoc1960/ASI-Forge.git
cd ASI-Forge
pip install -r requirements.txt

# Start services (requires Docker)
cd database && docker-compose up -d && ./start_api.sh &
cd ../cognition_base && docker-compose up -d && python rag_api.py &

# Run the interactive CLI
cd pipeline
python -m cli
```

## Creating a Specialization

1. Launch the CLI: `python -m cli`
2. Select `[N] Create New Specialization`
3. Provide:
   - **Name and description** of your research domain
   - **Initialization mode**: Seeded or Genesis
   - **Reference documents** (optional): Papers/docs to build the knowledge base

The system automatically generates:
- Domain-specific prompts for all 8 agents
- Validation constraints tailored to your domain
- Evaluation infrastructure
- Knowledge base from your documents + web search

See [docs/SPECIALIZATION_GUIDE.md](docs/SPECIALIZATION_GUIDE.md) for detailed documentation.

## Architecture

```
ASI-Forge
├── pipeline/                 # Core research pipeline
│   ├── evolve/              # Planner, Checker, Deduplication agents
│   ├── eval/                # Trainer, Debugger agents
│   ├── analyse/             # Analyzer agent
│   ├── database/            # Summarizer agent
│   └── specialization/      # Domain bootstrapping system
├── database/                # MongoDB + experiment storage
├── cognition_base/          # RAG knowledge base
└── specializations/         # Domain configurations
```

## Attribution

ASI-Forge is built on **ASI-Arch** by Liu et al. (GAIR-NLP, 2025). ASI-Arch demonstrated that LLMs can conduct end-to-end scientific research autonomously, discovering 106 novel linear attention architectures. ASI-Forge generalizes this capability to any research domain.

**Original Paper**: [AlphaGo Moment for Model Architecture Discovery](https://arxiv.org/abs/2507.18074)

**Original Repository**: [GAIR-NLP/ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch)

If you use ASI-Forge, please cite the original ASI-Arch paper:

```bibtex
@misc{liu2025alphagomomentmodelarchitecture,
      title={AlphaGo Moment for Model Architecture Discovery},
      author={Yixiu Liu and Yang Nan and Weixian Xu and Xiangkun Hu and Lyumanshan Ye and Zhen Qin and Pengfei Liu},
      year={2025},
      eprint={2507.18074},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.18074},
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
