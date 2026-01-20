# API Setup Guide for AEGIS

This guide explains how to securely configure API keys for AEGIS's knowledge augmentation system.

## 🔒 Security First

**NEVER** hardcode API keys in your code or commit them to version control!

AEGIS uses environment variables to keep your API keys secure.

---

## Quick Setup

### 1. Copy the Example File

```bash
cd /mnt/d/aegis
cp .env.example .env
```

### 2. Edit `.env` with Your API Keys

```bash
nano .env  # or use your favorite editor
```

### 3. Verify `.env` is in `.gitignore`

The `.env` file is already excluded from git, but double-check:

```bash
cat .gitignore | grep .env
```

You should see `.env` listed.

---

## API Keys You Need

### 🤖 Anthropic API (Optional - for advanced content processing)

**What it does**: Process and analyze web content using Claude

**Get your key**:
1. Go to: https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Copy the key (starts with `sk-ant-api03-...`)
4. **IMPORTANT**: Delete any exposed keys immediately!

**Add to `.env`**:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

**Note**: After the security incident, make sure to:
- Revoke any previously exposed keys
- Generate a fresh key
- Never share it again

---

### 🔍 Google Custom Search API (Optional - for real web search)

**What it does**: Perform real web searches instead of simulated results

**Setup Steps**:

1. **Get Google API Key**:
   - Go to: https://console.cloud.google.com/apis/credentials
   - Create a new project (if needed)
   - Click "Create Credentials" → "API Key"
   - Copy the API key
   - Enable "Custom Search API" for your project

2. **Create Custom Search Engine**:
   - Go to: https://programmablesearchengine.google.com/
   - Click "Add" to create a new search engine
   - Configure it to search the entire web
   - Copy the "Search Engine ID" (looks like: `a1b2c3d4e5f6g7h8i`)

**Add to `.env`**:
```bash
GOOGLE_SEARCH_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
```

---

### 📚 arXiv Academic Search (Free - No Key Required!)

**What it does**: Search academic papers on arXiv

**Setup**: None required! arXiv API is free and works automatically.

The system will use arXiv for academic searches by default.

---

## Testing Your Setup

### 1. Check Environment Variables are Loaded

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Check what's configured (without exposing keys)
if os.getenv('ANTHROPIC_API_KEY'):
    print("✓ Anthropic API key found")
else:
    print("✗ Anthropic API key not found")

if os.getenv('GOOGLE_SEARCH_API_KEY'):
    print("✓ Google Search API key found")
else:
    print("✗ Google Search API key not found")
```

### 2. Test Knowledge Augmentation System

```python
from core.agency.knowledge_augmentation import KnowledgeAugmentationSystem

# Initialize (will auto-load from .env)
knowledge_system = KnowledgeAugmentationSystem()

# Test academic search (uses free arXiv API)
results = knowledge_system.search_and_learn(
    query="neural architecture search",
    topic="ai_research",
    search_type="academic"
)

print(f"Found {len(results)} knowledge items")
```

### 3. Run AEGIS with Real APIs

```bash
# The system will automatically use real APIs if keys are present
python aegis_autonomous.py
```

Check the logs - you should see:
```
WebSearchEngine initialized with: Google Search, Anthropic
```

Instead of:
```
WebSearchEngine initialized with no API keys (using simulation mode)
```

---

## Fallback Behavior

AEGIS is designed to work even without API keys:

| API Missing | Behavior |
|-------------|----------|
| No Google API | Uses simulated web search results |
| No Anthropic API | Works normally (API is optional) |
| No keys at all | Fully functional with simulated data |

This allows you to:
- Test AEGIS without any API costs
- Develop and debug offline
- Add real APIs when ready for production

---

## API Usage and Costs

### Anthropic API
- **Pricing**: Pay per token (input + output)
- **Claude Sonnet**: ~$3 per million input tokens
- **Usage in AEGIS**: Optional content processing
- **Control**: Not used in current implementation (ready for future features)

### Google Custom Search API
- **Free Tier**: 100 queries/day
- **Paid**: $5 per 1,000 queries after free tier
- **Usage in AEGIS**: Each web search = 1 query
- **Typical usage**: 10-50 queries per autonomous session

### arXiv API
- **Cost**: FREE!
- **Limits**: ~1 request per 3 seconds (polite crawling)
- **Usage in AEGIS**: Academic searches only

---

## Security Best Practices

### ✅ DO:
- Store API keys in `.env` file
- Add `.env` to `.gitignore`
- Use separate keys for development and production
- Rotate keys periodically
- Set usage limits in API console
- Monitor API usage for anomalies

### ❌ DON'T:
- Commit API keys to git
- Share API keys in chat/email
- Hardcode keys in source code
- Use production keys for testing
- Give keys broader permissions than needed
- Leave unused keys active

---

## Revoking Compromised Keys

If you accidentally expose an API key:

### Anthropic:
1. Go to https://console.anthropic.com/settings/keys
2. Find the compromised key
3. Click "Delete"
4. Generate a new key
5. Update your `.env` file

### Google:
1. Go to https://console.cloud.google.com/apis/credentials
2. Find the compromised key
3. Click the key → "Delete"
4. Create a new key
5. Update your `.env` file

### Then:
```bash
# Clear any cached credentials
rm -rf ~/.cache/anthropic
rm -rf ~/.cache/google

# Restart AEGIS
python aegis_autonomous.py
```

---

## Advanced Configuration

### Custom API Endpoints

You can also specify custom endpoints in `.env`:

```bash
# Use a proxy or custom endpoint
ANTHROPIC_API_BASE_URL=https://your-proxy.com/v1
GOOGLE_SEARCH_BASE_URL=https://your-proxy.com/search
```

### Per-Environment Setup

```bash
# Development
cp .env.example .env.development
# Add dev keys to .env.development

# Production
cp .env.example .env.production
# Add prod keys to .env.production

# Load specific environment
export ENV=production
python aegis_autonomous.py
```

---

## Troubleshooting

### "WebSearchEngine initialized with no API keys"

**Cause**: `.env` file not found or empty

**Fix**:
```bash
# Check file exists
ls -la .env

# Check contents (careful not to expose keys!)
cat .env | grep -v "^#" | grep -v "^$" | wc -l  # Should show >0

# Verify python-dotenv is installed
pip show python-dotenv
```

### "Google Search API error: 403"

**Cause**: API key invalid or Custom Search API not enabled

**Fix**:
1. Verify key is correct in `.env`
2. Enable Custom Search API in Google Cloud Console
3. Check API usage limits

### "arXiv search failed"

**Cause**: Network issue or rate limiting

**Fix**:
- Check internet connection
- Wait a few seconds between searches
- arXiv API has rate limits (3 seconds between requests)

---

## Support

For issues with:
- **AEGIS setup**: Check `SETUP.md`
- **API integration**: Check `API_SETUP.md` (this file)
- **Security concerns**: Revoke keys immediately, then review this guide

---

**Remember**: When in doubt, regenerate your API keys. It's always better to be safe!
