# Quick Guide: Get Your Google Search Engine ID

## Step-by-Step:

### 1. Create a Programmable Search Engine

1. **Go to**: https://programmablesearchengine.google.com/
2. **Sign in** with your Google account
3. **Click the blue "Add" button**

### 2. Configure Your Search Engine

Fill in the form:
- **Search engine name**: `AEGIS Web Search` (or any name you want)
- **What to search**:
  - Toggle ON: **"Search the entire web"**
  - This is CRITICAL - you want to search ALL websites, not just specific ones
- Click **"Create"**

### 3. Get Your Search Engine ID

After creating, you'll see:
- A code that looks like: `a1b2c3d4e5f6g7h8i` (mix of letters and numbers)
- It's also labeled as "**Search engine ID**" or "**cx parameter**"
- **Copy this ID**

### 4. Update Your .env File

Open `/mnt/d/aegis/.env` and replace the placeholder:

```bash
GOOGLE_SEARCH_ENGINE_ID=YOUR_ACTUAL_ID_HERE
```

For example:
```bash
GOOGLE_SEARCH_ENGINE_ID=a1b2c3d4e5f6g7h8i
```

### 5. Restart AEGIS

```bash
python start_training.py
# or
python demo.py
```

## Verify It's Working

You should see in the logs:
```
✓ WebSearchEngine initialized with: Google Search, Anthropic
```

And when searching:
```
✓ Google Search returned 5 results
✓ Added 5 knowledge items from search
```

Instead of:
```
✗ No Google API key found, using simulated search
```

## Free Tier Limits

- **100 queries per day** for free
- After that: $5 per 1,000 queries
- Perfect for testing and development!

## Troubleshooting

### "Search engine ID not found"
- Make sure you copied the correct ID (it's alphanumeric)
- Check there are no extra spaces in your .env file

### "403 Forbidden" error
- Go to Google Cloud Console: https://console.cloud.google.com/
- Enable "Custom Search API" for your project
- Make sure your API key is valid

### Still getting simulated results?
```bash
# Test if environment variables are loaded:
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('API Key:', 'FOUND' if os.getenv('GOOGLE_SEARCH_API_KEY') else 'MISSING')
print('Engine ID:', 'FOUND' if os.getenv('GOOGLE_SEARCH_ENGINE_ID') else 'MISSING')
"
```

---

**Once configured, you'll get REAL search results and REAL insights!** 🎉
