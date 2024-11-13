from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import http.client
import os
import json
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Azure configuration
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
ai_vision_api_key = os.getenv("AZURE_AI_VISION_API_KEY")
ai_vision_region = os.getenv("AZURE_AI_VISION_REGION")
credential = AzureKeyCredential(key)

# Azure Search client
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# Function to vectorize the input image using Azure AI Vision
def get_image_vector(image_data, key, region):
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'model-version': '2023-04-15'
    }
    conn = http.client.HTTPSConnection(f'{region}.api.cognitive.microsoft.com')
    conn.request("POST", "/computervision/retrieval:vectorizeImage?api-version=2023-04-01-preview", image_data, headers)
    response = conn.getresponse()
    data = json.load(response)
    conn.close()
    
    if response.status != 200:
        raise Exception(f"Error in vectorizing image: {data.get('message', '')}")
    
    return data.get("vector")

# Helper function to resize and encode image in base64
def encode_image_base64(image_path, max_size=(300, 300)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Endpoint to handle image upload and perform similarity search
@app.route('/search', methods=['POST'])
def search_similar_images():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Read the uploaded image
    uploaded_image = request.files['image']
    image_data = uploaded_image.read()
    
    # Vectorize the uploaded image
    try:
        query_vector = get_image_vector(image_data, ai_vision_api_key, ai_vision_region)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Prepare the vectorized query
    vectorized_query = VectorizedQuery(
        vector=query_vector,
        k=2,  # Restrict results to top 2 images
        fields="image_vector"
    )

    # Perform vector search in Azure Search
    results = search_client.search(
        search_text=None,
        vector_queries=[vectorized_query],
        select=["description"]
    )
    
    # Collect and encode the top 2 similar images
    similar_images = []
    for result in results:
        image_path = os.path.join("images", result['description'])
        
        if os.path.exists(image_path):
            encoded_image = encode_image_base64(image_path)
            similar_images.append({
                "description": result['description'],
                "image_base64": encoded_image
            })
        
        if len(similar_images) == 2:  # Limit to top 2 results
            break
    
    # Return the similar images in JSON format
    return jsonify(similar_images), 200

if __name__ == '__main__':
    app.run(debug=True)
