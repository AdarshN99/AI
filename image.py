import os
import json
import requests
import http.client
import urllib.parse
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery  # Update to correct class
import base64
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables
load_dotenv()
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
aiVisionApiKey = os.getenv("AZURE_AI_VISION_API_KEY")
aiVisionRegion = os.getenv("AZURE_AI_VISION_REGION")
aiVisionEndpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")
IMAGES_DIR = "images"  # Directory where images are stored locally

app = Flask(__name__)
credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

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

    except (requests.exceptions.Timeout, http.client.HTTPException) as e:
        print(f"Timeout/Error for {image_path}. Retrying...")
        raise

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route("/search_image", methods=["POST"])
def search_image():
    try:
        # Get image path from request body
        image_name = request.json.get("image_path")
        image_path = os.path.join(IMAGES_DIR, image_name)  # Full path in 'images' folder

        # Get vector for query image
        query_vector = get_image_vector(image_path, aiVisionApiKey, aiVisionRegion)

        # Create the vectorized query
        vectorized_query = VectorizedQuery(
            vector=query_vector,
            fields="image_vector"
        )

        # Execute the vector search query with 'top' parameter to restrict to top 2 results
        search_results = search_client.search(vectorized_query, top=2)

        # Format and return results with base64-encoded images
        output = []
        for result in search_results:
            similar_image_path = os.path.join(IMAGES_DIR, result["description"])
            encoded_image = encode_image_to_base64(similar_image_path)
            output.append({
                "description": result["description"],
                "image_base64": encoded_image
            })

        return jsonify({"results": output}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
