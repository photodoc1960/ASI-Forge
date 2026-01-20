# Quick Start: API Configuration

Get AEGIS working with real web search in 5 minutes!

---

## Option 1: No APIs (Easiest - Start Here!)

AEGIS works perfectly without any API keys using simulated search:

```bash
python aegis_autonomous.py
```

**Perfect for**:
- Testing and development
- Learning how AEGIS works
- Offline usage
- No cost experimentation

---

## Option 2: Free arXiv Only (Academic Papers)

Get real academic papers with **zero setup**:

```python
from core.agency.knowledge_augmentation import KnowledgeAugmentationSystem

knowledge = KnowledgeAugmentationSystem()

# Search arXiv (free, no key needed!)
results = knowledge.search_and_learn(
    query="neural architecture search",
    topic="ai_research",
    search_type="academic"
)

print(f"Found {len(results)} real papers from arXiv!")
```

**Perfect for**:
- Academic research
- Free real data
- Paper discovery

---

## Option 3: Full Setup (Google + Anthropic)

### Step 1: Create `.env` File

```bash
cp .env.example .env
nano .env  # or use your editor
```

### Step 2: Add API Keys

```bash
# Get keys from:
# - Anthropic: https://console.anthropic.com/settings/keys
# - Google: https://console.cloud.google.com/apis/credentials

ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
GOOGLE_SEARCH_API_KEY=your-google-key-here
GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id-here
```

### Step 3: Run AEGIS

```bash
python aegis_autonomous.py
```

You should see:
```
WebSearchEngine initialized with: Google Search, Anthropic
```

**Perfect for**:
- Production use
- Real web search
- Advanced features

---

## Security Checklist

- [ ] Copied `.env.example` to `.env`
- [ ] Added keys to `.env` (not source code!)
- [ ] Verified `.env` is in `.gitignore`
- [ ] Never committed `.env` to git
- [ ] Revoked any previously exposed keys

---

## Troubleshooting

### "No API keys found"
✅ **This is normal!** AEGIS works fine without keys using simulation mode.

### "Want real web search"
📖 See `API_SETUP.md` for detailed instructions on getting Google API keys.

### "Just want academic papers"
🎓 arXiv works automatically! Just use `search_type="academic"` - no setup needed.

### "Accidentally exposed a key"
🚨 **Act now**:
1. Go to API console and DELETE the key immediately
2. Generate a new key
3. Update your `.env` file
4. Never share keys in chat/email/code

---

## Quick Test

```python
# Test your setup
from core.agency.knowledge_augmentation import KnowledgeAugmentationSystem
import logging

logging.basicConfig(level=logging.INFO)
knowledge = KnowledgeAugmentationSystem()

# This will use whatever APIs you have configured
results = knowledge.search_and_learn(
    query="transformer models",
    topic="ml",
    search_type="academic"
)

print(f"Success! Found {len(results)} results")
```

---

**Full docs**: See `API_SETUP.md` for complete instructions.

**Security**: See `SECURITY_UPDATE.md` for security details.
