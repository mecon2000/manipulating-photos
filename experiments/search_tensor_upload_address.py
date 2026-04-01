from ddgs import DDGS
import json

def search():
    with DDGS() as ddgs:
        results = [r for r in ddgs.text("python tams GetResourceUploadAddress tensor.art", max_results=5)]
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    search()
