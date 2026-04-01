from ddgs import DDGS
import json

def search():
    with DDGS() as ddgs:
        results = [r for r in ddgs.text("tensor.art api upload image resource id", max_results=5)]
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    search()
