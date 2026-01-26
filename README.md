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

---

## Prerequisites

- **Docker** and **Docker Compose** - For running MongoDB and OpenSearch services
- **Python 3.10+** - The framework requires Python 3.10 or higher
- **CUDA** (optional) - Required for GPU-accelerated training in ML-focused specializations
- **OpenAI API Key** - The agents use OpenAI's API for LLM capabilities

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/photodoc1960/ASI-Forge.git
cd ASI-Forge
```

### 2. Create Python Environment

```bash
# Using conda (recommended)
conda create -n asi-forge python=3.10
conda activate asi-forge

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key-here
```

---

## Starting Services

ASI-Forge requires two backend services: **MongoDB** (experiment storage) and **OpenSearch** (RAG knowledge base).

### 1. Start MongoDB

```bash
cd database
docker-compose up -d
```

This starts:
- **MongoDB** on port `27018` (mapped from container's 27017)
- **Mongo Express** (web UI) on port `8081` - access at http://localhost:8081
  - Username: `admin`
  - Password: `admin123`

Verify MongoDB is running:
```bash
docker-compose ps
# Should show mongodb and mongo-express as "Up"
```

### 2. Start OpenSearch (RAG Service)

```bash
cd cognition_base
docker-compose up -d
```

This starts:
- **OpenSearch** on port `9200`
- **OpenSearch Dashboards** on port `5601` - access at http://localhost:5601

Verify OpenSearch is running:
```bash
curl http://localhost:9200
# Should return cluster info JSON
```

### 3. Start the RAG API

In a separate terminal:
```bash
cd cognition_base
python rag_api.py
```

This starts the RAG API server on port `5000`.

---

## Running ASI-Forge

### Launch the Interactive CLI

```bash
cd pipeline
python -m cli
```

You'll see the main menu:

```
╔══════════════════════════════════════════════════════════════╗
║           ASI-Forge: Autonomous Research Framework           ║
╠══════════════════════════════════════════════════════════════╣
║  Available Specializations:                                  ║
║                                                              ║
║  [1] Linear Attention Architectures                          ║
║      └─ 106 experiments | Best score: 0.847                  ║
║                                                              ║
║  [N] Create New Specialization                               ║
║  [R] Run Pipeline                                            ║
║  [Q] Quit                                                    ║
╚══════════════════════════════════════════════════════════════╝
```

### Creating a New Specialization

1. Select `[N] Create New Specialization`
2. Provide:
   - **Name**: Short identifier (e.g., "protein-folding")
   - **Description**: Detailed description of your research domain
   - **Initialization mode**:
     - `Seeded`: Point to an existing codebase to evolve
     - `Genesis`: Generate initial code from scratch
   - **Reference documents** (optional): Papers/docs to build the knowledge base

The system automatically generates:
- Domain-specific prompts for all 8 agents
- Validation constraints tailored to your domain
- Evaluation infrastructure (training/testing scripts)
- Knowledge base from your documents + web search

### Running Experiments

1. Select a specialization from the menu
2. Select `[R] Run Pipeline`
3. Enter the number of experiments to run

The pipeline will:
1. Sample a program from the database
2. Generate an evolved version using the Planner agent
3. Validate the code with the Checker agent
4. Evaluate the evolved code (training/testing)
5. Debug failures with the Debugger agent
6. Analyze results with the Analyzer agent
7. Store successful experiments in the database

---

## Architecture

```
ASI-Forge/
├── pipeline/                 # Core research pipeline
│   ├── evolve/              # Planner, Checker, Deduplication agents
│   ├── eval/                # Trainer, Debugger agents
│   ├── analyse/             # Analyzer agent
│   ├── database/            # Summarizer agent
│   ├── specialization/      # Domain bootstrapping system
│   │   ├── agents/          # Bootstrapper agents
│   │   ├── schema.py        # Data models
│   │   ├── manager.py       # Specialization CRUD
│   │   └── bootstrapper.py  # Orchestrator
│   └── cli.py               # Interactive CLI
├── database/                # MongoDB + experiment storage
│   ├── docker-compose.yml   # MongoDB containers
│   └── mongodb_api.py       # Database API
├── cognition_base/          # RAG knowledge base
│   ├── docker-compose.yml   # OpenSearch containers
│   ├── rag_api.py          # RAG API server
│   └── rag_service.py      # OpenSearch interface
└── specializations/         # Domain configurations
    └── <domain>/
        ├── config.json      # Domain configuration
        ├── prompts/         # Generated prompts
        ├── infrastructure/  # Evaluation scripts
        └── seed/            # Initial codebase
```

---

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
cd database
docker-compose ps

# View MongoDB logs
docker-compose logs mongodb

# Restart services
docker-compose down
docker-compose up -d
```

### OpenSearch Connection Issues

```bash
# Check if OpenSearch is running
cd cognition_base
docker-compose ps

# Check cluster health
curl http://localhost:9200/_cluster/health

# View logs
docker-compose logs opensearch
```

### Python Import Errors

Make sure you're in the correct conda/venv environment:
```bash
conda activate asi-forge
# or
source venv/bin/activate
```

---

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

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
