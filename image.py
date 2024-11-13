import os
import json
import requests
import http.client
import urllib.parse
from flask import Flask, request, jsonify
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery

# Load environment variables from .env
load_dotenv()

# Azure configurations
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
aiVisionApiKey = os.getenv("AZURE_AI_VISION_API_KEY")
aiVisionRegion = os.getenv("AZURE_AI_VISION_REGION")

# Initialize Flask app
app = Flask(__name__)

# Set up Azure Search Client
credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# Absolute path for the 'images' folder in the project
PROJECT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")

# Function to get image vector using Azure AI Vision
@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_image_vector(image_path, key, region):
    headers = {
        'Ocp-Apim-Subscription-Key': key,
    }

    params = urllib.parse.urlencode({
        'model-version': '2023-04-15',
    })

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
        conn.request("POST", "/computervision/retrieval:vectorizeImage?api-version=2023-04-01-preview&%s" % params, body, headers)
        response = conn.getresponse()
        data = json.load(response)
        conn.close()

        if response.status != 200:
            raise Exception(f"Error processing image {image_path}: {data.get('message', '')}")

        return data.get("vector")

    except (requests.exceptions.Timeout, http.client.HTTPException) as e:
        print(f"Timeout/Error for {image_path}. Retrying...")
        raise

@app.route("/search_image", methods=["POST"])
def search_image():
    try:
        # Get image path and 'k' value from the request body
        image_name = request.json.get("image_path")
        image_path = os.path.join(IMAGES_DIR, image_name)  # Full path in 'images' folder
        k = request.json.get("k", 3)  # Default value of k is 3

        # Get the vector for the query image
        query_vector = get_image_vector(image_path, aiVisionApiKey, aiVisionRegion)

        # Define the VectorQuery
        vector_query = VectorQuery(vector=query_vector, k=k, fields=["image_vector"])

        # Perform the search
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],  # Corrected keyword
            select=["description"]
        )

        # Format the results
        output = []
        for result in results:
            output.append({
                "description": result["description"]
            })

        return jsonify({"results": output}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
