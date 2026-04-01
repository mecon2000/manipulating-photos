import fal_client

# Not really easy to list models from client
# I'll try common names
models = ["fal-ai/inpainting", "fal-ai/sdxl-inpainting", "fal-ai/fovea-inpainting", "fal-ai/lama"]
for m in models:
    try:
        print(f"Testing {m}")
        # dummy call
        res = fal_client.subscribe(m, {"prompt": "test"}, with_logs=True)
        print(f"Found {m}")
    except Exception as e:
        print(f"Not found {m}: {e}")
