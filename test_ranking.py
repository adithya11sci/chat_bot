"""Diagnose & test ranking detection + full API."""
import urllib.request, json, sys

BASE = "http://localhost:8000/chat"

tests = [
    ("which book has the higgest average_rating", "average_rating"),
    ("which has the higgest rating", "average_rating"),
    ("which book has more pages", "num_pages"),
    ("Which book is most popular among readers?", "ratings_count"),
    ("show me the top rated books", "average_rating"),
    ("what is the oldest book in the dataset", "published_year"),
]

results = []
for question, key_col in tests:
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        BASE, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        results.append(f"ERROR for '{question}': {e}")
        continue

    lines = []
    lines.append("=" * 60)
    lines.append(f"Q: {question}")
    lines.append(f"A: {result['response'][:250]}")
    lines.append(f"Metadata count: {len(result.get('metadata', []))}")
    for i, b in enumerate(result.get("metadata", [])[:3]):
        title = b.get("title", "?")
        val = b.get(key_col, "?")
        lines.append(f"  {i+1}. {title}  |  {key_col}: {val}")
    lines.append("")
    results.append("\n".join(lines))

output = "\n".join(results)
with open("test_output.md", "w", encoding="utf-8") as f:
    f.write(output)
print("Done! Check test_output.md")
