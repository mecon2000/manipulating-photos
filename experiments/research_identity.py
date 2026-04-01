import sys
import os
sys.path.append(os.getcwd())
try:
    from googlesearch import search
    query = "Stable Diffusion maintain face identity consistency best practices 2025 2026"
    for j in search(query, num=10, stop=10, pause=2):
        print(j)
except ImportError:
    print("googlesearch-python not installed")
