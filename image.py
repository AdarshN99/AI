import os
import json
import requests
import http.client, urllib.parse
import base64
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchField, SearchFieldDataType
from azure.search.documents.models import VectorSearch, VectorizedQuery
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

# Set up Azure Cognitive Search client
credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# Set up Flask app
app = Flask(__name__)

# Path to images folder in your project
FILE_PATH = 'images'

# Sanitize file names to conform to Azure Cognitive Search ID constraints
def sanitize_id(filename):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

# Function to get image vector using Azure Vision API
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

# Function to convert an image to base64 encoding
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Upload images and vectors to Azure Search
def upload_images_to_search():
    files = os.listdir(FILE_PATH)
    image_embeddings = {}

    for file in files:
        file_path = os.path.join(FILE_PATH, file)
        image_embeddings[file] = get_image_vector(file_path, aiVisionApiKey, aiVisionRegion)

    documents = []
    for counter, file in enumerate(files):
        sanitized_id = sanitize_id(file)
        document = {
            "id": sanitized_id,
            "description": file,
            "image_vector": image_embeddings[file]
        }
        documents.append(document)

    # Upload documents to Azure Search
    result = search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents successfully.")

@app.route('/search', methods=['POST'])
def search_similar_images():
    file = request.files.get('image')
    
    if not file:
        return jsonify({"error": "No image file provided"}), 400
    
    # Save the uploaded file temporarily
    uploaded_image_path = os.path.join('uploads', file.filename)
    file.save(uploaded_image_path)

    # Get vector for the uploaded image
    query_vector = get_image_vector(uploaded_image_path, aiVisionApiKey, aiVisionRegion)

    # Create VectorizedQuery for similarity search
    vectorized_query = VectorizedQuery(
        kind="vector",
        vector=query_vector,
        k_nearest_neighbors=2,  # Limit to top 2 similar images
        fields="image_vector",  # Field in the index that stores image vectors
        exhaustive=True,
    )

    # Perform the search using VectorizedQuery
    results = search_client.search(
        search_text=None, 
        vector_queries=[vectorized_query],
        select=["description"]
    )

    # Return similar images as base64 encoded images
    similar_images = []
    for result in results:
        image_path = os.path.join(FILE_PATH, result["description"])
        encoded_image = image_to_base64(image_path)  # Convert image to base64
        similar_images.append({
            "image_name": result["description"],
            "image_base64": encoded_image
        })
    
    return jsonify({"similar_images": similar_images})

if __name__ == "__main__":
    # Upload images to Azure Cognitive Search initially
    upload_images_to_search()

    # Run the Flask app
    app.run(debug=True)
