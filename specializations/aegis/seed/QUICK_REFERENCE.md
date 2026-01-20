# AEGIS Quick Reference Card

## Starting AEGIS

```bash
./run_aegis.sh          # Launcher with menu
python aegis_autonomous.py   # Direct start
```

---

## Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `goals` | Show detailed agent goals |
| `pending` | Show pending approval requests |
| `think` | Make agent decide an action (with manual execution) |
| `ask <question>` | Ask agent a question (triggers web search) |
| `tell <info>` | Tell agent something (adds to knowledge base) |
| `approve <id>` | Approve a pending request |
| `reject <id>` | Reject a pending request |
| `status` | Show system status |
| `knowledge` | Show knowledge base stats |
| `help` | Show all commands |
| `quit` | Exit |

---

## Understanding Agent Behavior

### Normal Behavior ✅
- **Idle most of the time** (safety feature)
- **Proposes improvements every 10 iterations** in autonomous mode
- **Generates abstract goals** on startup
- **Waits for approval** before self-modifying

### Getting More Activity

**Option 1: Ask Questions**
```
>>> ask What is neural architecture search?
>>> ask How does attention work in transformers?
>>> ask What are the latest advances in meta-learning?
```

**Option 2: Approve Proposals**
```
>>> pending
>>> approve <request-id>
```

**Option 3: Use Think Command**
```
>>> think
# Shows what agent wants to do
# You choose to execute or not
```

---

## Goal Types

| Type | What Agent Does |
|------|----------------|
| `knowledge_acquisition` | Searches web or asks questions |
| `self_improvement` | Proposes architecture changes |
| `exploration` | Searches for new information |
| `understanding` | Builds mental models |
| `capability_improvement` | Tries to improve skills |

---

## Approval Workflow

### 1. See Pending Requests
```bash
>>> pending
```

### 2. Review Details
- **Title**: What's changing
- **Description**: How it's changing
- **Risk Assessment**: Safety analysis
- **Reversibility**: Can it be undone?

### 3. Approve or Reject
```bash
>>> approve 4f7fe7e1-e6f3-40a2-b174-0cc3e9af0841
# OR
>>> reject 4f7fe7e1-e6f3-40a2-b174-0cc3e9af0841
```

---

## Example Session

```bash
# Start interactive mode
python aegis_autonomous.py
# Choose 2

# See what agent wants
>>> goals

# Check for pending approvals
>>> pending

# Guide the agent
>>> ask What is hierarchical reasoning?
>>> ask What are neural architecture search methods?

# Check what it learned
>>> knowledge

# Make it think
>>> think
# Execute if action makes sense

# Approve any improvements
>>> pending
>>> approve <id>

# Check status
>>> status
```

---

## API Configuration (Optional)

Create `.env` file for real web search:

```bash
cp .env.example .env
nano .env
```

Add:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key
GOOGLE_SEARCH_API_KEY=your-google-key
GOOGLE_SEARCH_ENGINE_ID=your-search-id
```

**Note**: arXiv academic search works FREE without any API keys!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Agent always idle | Ask questions or approve proposals |
| No pending approvals | Run autonomous mode longer (20+ iterations) |
| Want more activity | Use interactive mode and guide with questions |
| API errors | Check `.env` file or use simulation mode (works fine!) |

---

## Safety Notes

### What Requires Approval ✅
- Code generation
- Architecture changes
- Capability deployment

### What Doesn't Require Approval ✅
- Web searches
- Knowledge acquisition
- Goal generation
- Thinking/decision-making

---

## Files

| File | Purpose |
|------|---------|
| `aegis_autonomous.py` | Main autonomous system |
| `USER_GUIDE.md` | Detailed usage guide (READ THIS!) |
| `API_SETUP.md` | API configuration guide |
| `QUICK_REFERENCE.md` | This file |
| `.env.example` | API key template |

---

## Quick Tips

1. **Start with interactive mode** to understand behavior
2. **Use `goals` and `pending` frequently**
3. **Ask questions to activate the agent**
4. **Approve proposals to let it evolve**
5. **The "idle" state is normal** (safety by design)
6. **No API keys needed** (simulation mode works great!)

---

**For detailed explanations**: See `USER_GUIDE.md`
