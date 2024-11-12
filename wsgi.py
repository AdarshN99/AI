from flask import Flask, request, jsonify
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure_openai import *  # Make sure azure_openai has `create_prompt` and `generate_answer`
from config import *        # Ensure config has `searchservice`, `searchkey`, and `index`

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search_document():
    # Retrieve the user's query
    user_input = request.json.get("question", "What is Diploblastic and Triploblastic Organisation?")
    
    # Define Azure Search service parameters
    service_name = searchservice
    key = searchkey
    endpoint = f"https://{searchservice}.search.windows.net/"
    index_name = index

    azure_credential = AzureKeyCredential(key)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=azure_credential)

    # Define search fields
    KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT", "content")
    KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY", "category")
    KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE", "sourcepage")
    
    exclude_category = None
    filter_condition = f"category ne '{exclude_category.replace("'", "''")}'" if exclude_category else None
    
    # Search query to Azure Search
    results = search_client.search(
        user_input, 
        filter=filter_condition,
        query_type=QueryType.SEMANTIC, 
        query_language="en-us", 
        query_speller="lexicon", 
        semantic_configuration_name="default", 
        top=3
    )

    # Process results
    content = "\n".join([f"{doc[KB_FIELDS_SOURCEPAGE]}: {doc[KB_FIELDS_CONTENT].replace('\n', '').replace('\r', '')}" for doc in results])
    references = list(set(doc[KB_FIELDS_SOURCEPAGE] for doc in results))

    # Prepare prompt and generate answer using OpenAI
    conversation = [{"role": "system", "content": "Assistant is a great language model formed by OpenAI."}]
    prompt = create_prompt(content, user_input)
    conversation.append({"role": "assistant", "content": prompt})
    conversation.append({"role": "user", "content": user_input})
    reply = generate_answer(conversation)

    # Return the response as JSON
    response = {
        "answer": reply,
        "references": references
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
