from ddgs import DDGS
import json

def search():
    with DDGS() as ddgs:
        results = [r for r in ddgs.text("github Tensor-Art ComfyUI_TENSOR_ART TA_UploadImage.py", max_results=2)]
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    search()
