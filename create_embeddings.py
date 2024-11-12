# create_embeddings.py
import os
import json
import requests
import http.client
import urllib.parse
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# Azure Vision API details
aiVisionApiKey = os.getenv("AZURE_AI_VISION_API_KEY")
aiVisionRegion = os.getenv("AZURE_AI_VISION_REGION")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_image_vector(image_path, key, region):
    headers = {'Ocp-Apim-Subscription-Key': key}
    params = urllib.parse.urlencode({'model-version': '2023-04-15'})

    try:
        if image_path.startswith(('http://', 'https://')):
            headers['Content-Type'] = 'application/json'
            body = json.dumps({"url": image_path})
        else:
            headers['Content-Type'] = 'application/octet-stream'
            with open(image_path, "rb") as filehandler:
                image_data = filehandler.read()
                body = image_data

        conn = http.client.HTTPSConnection(f'{region}.api.cognitive.microsoft.com', timeout=3)
        conn.request("POST", f"/computervision/retrieval:vectorizeImage?api-version=2023-04-01-preview&{params}", body, headers)
        response = conn.getresponse()
        data = json.load(response)
        conn.close()

        if response.status != 200:
            raise Exception(f"Error processing image {image_path}: {data.get('message', '')}")

        return data.get("vector")

    except (requests.exceptions.Timeout, http.client.HTTPException):
        raise

# Directory with images
FILE_PATH = 'images'
FILES = os.listdir(FILE_PATH)
image_embeddings = {}

# Generate embeddings
for file in FILES:
    image_embeddings[file] = get_image_vector(os.path.join(FILE_PATH, file), aiVisionApiKey, aiVisionRegion)

# Prepare data for Azure Search index
input_data = [{"id": str(i), "description": file, "image_vector": image_embeddings[file]} for i, file in enumerate(FILES)]

# Save embeddings to a JSON file
os.makedirs('output', exist_ok=True)
with open("output/docVectors.json", "w") as f:
    json.dump(input_data, f)
