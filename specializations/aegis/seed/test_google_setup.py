#!/usr/bin/env python
"""Quick test to verify Google Search API setup"""

from dotenv import load_dotenv
import os

print("=" * 70)
print("GOOGLE SEARCH API SETUP TEST")
print("=" * 70)

# Load environment variables
load_dotenv()

# Check API Key
api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
if api_key:
    print(f"✓ GOOGLE_SEARCH_API_KEY: Found ({len(api_key)} characters)")
    if api_key.startswith('AIza'):
        print("  ✓ Format looks correct (starts with 'AIza')")
    else:
        print("  ⚠ Format might be wrong (should start with 'AIza')")
else:
    print("✗ GOOGLE_SEARCH_API_KEY: NOT FOUND")
    print("  Add it to your .env file")

print()

# Check Engine ID
engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
if engine_id:
    print(f"✓ GOOGLE_SEARCH_ENGINE_ID: Found ({len(engine_id)} characters)")
    if 'your_search_engine' in engine_id.lower() or 'paste' in engine_id.lower():
        print("  ✗ This is still a PLACEHOLDER!")
        print("  You need to replace it with your actual Search Engine ID")
        print("  See: GOOGLE_SETUP_QUICK.md for instructions")
    elif len(engine_id) > 5:
        print("  ✓ Format looks valid (not a placeholder)")
    else:
        print("  ⚠ Seems too short, might be incorrect")
else:
    print("✗ GOOGLE_SEARCH_ENGINE_ID: NOT FOUND")
    print("  Add it to your .env file")

print()
print("=" * 70)

# Try a test search if both are configured
if api_key and engine_id and 'your_search_engine' not in engine_id.lower():
    print("TESTING REAL GOOGLE SEARCH...")
    print("=" * 70)

    try:
        from core.agency.knowledge_augmentation import WebSearchEngine

        search_engine = WebSearchEngine()

        print(f"\n✓ Search engine initialized")
        print(f"  Has API key: {bool(search_engine.google_api_key)}")
        print(f"  Has Engine ID: {bool(search_engine.google_search_engine_id)}")

        print(f"\nPerforming test search: 'machine learning'...")
        results = search_engine.search("machine learning", max_results=3)

        print(f"\n✓ Search completed!")
        print(f"  Results found: {len(results)}")

        if results:
            print(f"\n  Sample result:")
            print(f"    Title: {results[0].title}")
            print(f"    URL: {results[0].url}")
            print(f"    Snippet: {results[0].snippet[:100]}...")

            # Check if it's real or simulated
            if 'example.com' in results[0].url:
                print(f"\n  ⚠ WARNING: Getting SIMULATED results!")
                print(f"    Your Search Engine ID might be incorrect")
            else:
                print(f"\n  ✅ SUCCESS! Getting REAL search results!")
                print(f"    Your Google Search API is working correctly!")
        else:
            print(f"\n  ⚠ No results returned")

    except Exception as e:
        print(f"\n✗ Error during test search: {e}")
        print(f"  This might be a configuration issue")

else:
    print("SKIPPING TEST SEARCH")
    print("  Configure both API key and Engine ID first")
    print("  Then run this script again")

print("=" * 70)
