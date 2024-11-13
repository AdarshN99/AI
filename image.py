import os
import json
import base64
from flask import Flask, jsonify
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from tenacity import retry, stop_after_attempt, wait_fixed
import urllib.parse
import http.client
import requests

# Load environment variables
load_dotenv()

# Azure Cognitive Search credentials and parameters
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
aiVisionApiKey = os.getenv("AZURE_AI_VISION_API_KEY")
aiVisionRegion = os.getenv("AZURE_AI_VISION_REGION")
aiVisionEndpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")

# Azure Cognitive Search client
credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# Flask app setup
app = Flask(__name__)

# Image upload folder
FILE_PATH = 'images'


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_image_vector(image_path, key, region):
    """Function to get vector for the image."""
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


@app.route('/upload_all_images', methods=['POST'])
def upload_all_images():
    """Uploads all images in the images folder, vectorizes them, and pushes to Azure Cognitive Search."""
    image_files = os.listdir(FILE_PATH)

    # Only process image files
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Collect all documents to upload to Azure Search
    documents = []

    for image_filename in image_files:
        image_path = os.path.join(FILE_PATH, image_filename)
        try:
            # Get the vector for the image
            image_vector = get_image_vector(image_path, aiVisionApiKey, aiVisionRegion)
            
            # Prepare document to upload
            doc = {
                "id": image_filename,  # Using filename as the unique ID
                "description": image_filename,
                "image_vector": image_vector
            }
            documents.append(doc)
        
        except Exception as e:
            return jsonify({"error": f"Error processing image {image_filename}: {str(e)}"}), 500

    # Upload all the documents to Azure Search
    if documents:
        try:
            search_client.upload_documents(documents=documents)
            return jsonify({"message": f"Successfully uploaded {len(documents)} images to Azure Cognitive Search."}), 200
        except Exception as e:
            return jsonify({"error": f"Error uploading images to Azure Search: {str(e)}"}), 500
    else:
        return jsonify({"message": "No images to upload."}), 200


@app.route('/search_image', methods=['POST'])
def search_image():
    """Search for similar images in Azure Cognitive Search."""
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']
    image_path = os.path.join(FILE_PATH, image.filename)
    image.save(image_path)

    try:
        image_vector = get_image_vector(image_path, aiVisionApiKey, aiVisionRegion)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    vector_query = VectorizedQuery(
        kind="vector",
        k_nearest_neighbors=2,
        fields="image_vector",
        vector=image_vector,
        exhaustive=True
        
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["description"]
    )

    similar_images = []
    for result in results:
        image_path = os.path.join(FILE_PATH, result["description"])
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            encoded_image = base64.b64encode(img_data).decode('utf-8')

        similar_images.append({
            "description": result["description"],
            "image_data": encoded_image
        })

    return jsonify({"similar_images": similar_images}), 200


if __name__ == '__main__':
    app.run(debug=True)
