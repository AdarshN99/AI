import os
import json
import logging
from semantic_kernel import Kernel
from semantic_kernel.ai.openai.services.azure_openai import AzureOpenAIService
from semantic_kernel.memory.azure_cognitive_search import AzureCognitiveSearchMemory
from semantic_kernel.core.memory.memory_record import MemoryRecord
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Initialize Semantic Kernel
sk = Kernel()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Azure Cognitive Search configuration
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME_TEXT = os.environ["INDEX_NAME_TEXT"]
INDEX_NAME_IMAGE = os.environ["INDEX_NAME_IMAGE"]

# Azure OpenAI configuration
OPENAI_DEPLOYMENT_ID = os.environ["OPENAI_DEPLOYMENT_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ENDPOINT = os.environ["OPENAI_ENDPOINT"]

# Initialize Azure Cognitive Search Memory
search_memory = AzureCognitiveSearchMemory(
    endpoint=SEARCH_ENDPOINT,
    api_key=SEARCH_API_KEY,
    index_name=INDEX_NAME_TEXT,
)

# Initialize Azure OpenAI Service
sk.add_service(
    "azure_openai",
    AzureOpenAIService(
        deployment_id=OPENAI_DEPLOYMENT_ID,
        api_key=OPENAI_API_KEY,
        endpoint=OPENAI_ENDPOINT,
    ),
)


@app.route("/text_search", methods=["POST"])
def text_search():
    """Perform text search using Azure Cognitive Search."""
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "Question is required"}), 400

    try:
        # Perform search in Azure Cognitive Search
        results = search_memory.search(user_input, top=3)
        search_results = [
            {
                "source": result.metadata["sourcepage"],
                "content": result.text,
            }
            for result in results
        ]
        return jsonify({"results": search_results}), 200
    except Exception as e:
        logging.error(f"Error in text_search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload_images", methods=["POST"])
def upload_images():
    """Upload image embeddings to Azure Cognitive Search."""
    image_dir = os.getenv("FILE_PATH_IMG", "./images")
    files = os.listdir(image_dir)
    documents = []

    for file in files:
        file_path = os.path.join(image_dir, file)
        try:
            # Convert image to embeddings
            vector = sk.services["azure_openai"].vectorize_image(file_path)
            sanitized_id = re.sub(r"[^a-zA-Z0-9_-]", "_", file)

            # Create memory record
            memory_record = MemoryRecord(
                id=sanitized_id,
                metadata={"description": file},
                vector=vector,
            )
            documents.append(memory_record)

        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")
            continue

    if documents:
        search_memory.save_batch(documents)
        return jsonify({"message": f"Uploaded {len(documents)} documents."}), 200
    else:
        return jsonify({"message": "No valid images to upload."}), 400


@app.route("/image_search", methods=["POST"])
def image_search():
    """Search for similar images using vectorized queries."""
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Image file is required"}), 400

        # Save uploaded file temporarily
        temp_path = os.path.join("./uploads", file.filename)
        file.save(temp_path)

        # Generate image vector
        vector = sk.services["azure_openai"].vectorize_image(temp_path)

        # Perform vector search
        results = search_memory.vector_search(vector, top=5)
        similar_images = [
            {"image_name": result.metadata["description"]} for result in results
        ]
        return jsonify({"similar_images": similar_images}), 200

    except Exception as e:
        logging.error(f"Error in image_search: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
