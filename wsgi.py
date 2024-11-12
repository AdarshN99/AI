from flask import Flask, request, jsonify
import os
import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure_openai import *  # Ensure this has `create_prompt` and `generate_answer` functions
from config import *        # Ensure config has `searchservice`, `searchkey`, and `index`

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/search', methods=['POST'])
def search_document():
    try:
        # Validate and retrieve the user’s input question
        user_input = request.json.get("question")
        if not user_input:
            return jsonify({"error": "Missing 'question' parameter in request body"}), 400

        # Initialize Azure Search Service parameters
        service_name = searchservice
        key = searchkey
        endpoint = f"https://{searchservice}.search.windows.net/"
        index_name = index

        azure_credential = AzureKeyCredential(key)
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=azure_credential)

        # Define field mappings for Azure Search
        KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT", "content")
        KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY", "category")
        KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE", "sourcepage")
        exclude_category = None
        filter_condition = f"category ne '{exclude_category.replace("'", "''")}'" if exclude_category else None

        # Search query to Azure Search with error handling
        try:
            results = search_client.search(
                user_input,
                filter=filter_condition,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name="default",
                top=3
            )
        except HttpResponseError as e:
            logging.error(f"Azure Search API error: {e.message}")
            return jsonify({"error": "Failed to retrieve search results from Azure Search API"}), 500
        except ServiceRequestError as e:
            logging.error(f"Service request error: {e}")
            return jsonify({"error": "Connection issue with Azure Search API"}), 500

        # Process results
        content = "\n".join([f"{doc[KB_FIELDS_SOURCEPAGE]}: {doc[KB_FIELDS_CONTENT].replace('\n', '').replace('\r', '')}" for doc in results])
        references = list(set(doc[KB_FIELDS_SOURCEPAGE] for doc in results))

        # Use OpenAI to generate a response based on the content and user input
        try:
            conversation = [{"role": "system", "content": "Assistant is a great language model formed by OpenAI."}]
            prompt = create_prompt(content, user_input)
            conversation.append({"role": "assistant", "content": prompt})
            conversation.append({"role": "user", "content": user_input})
            reply = generate_answer(conversation)
        except Exception as e:
            logging.error(f"Error generating response with OpenAI: {e}")
            return jsonify({"error": "Failed to generate answer from OpenAI model"}), 500

        # Return the response with answer and references
        response = {
            "answer": reply,
            "references": references
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
