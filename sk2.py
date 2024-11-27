import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.openai.chat_completion import AzureChatCompletion
from semantic_kernel.chat_history import ChatHistory

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Initialize Semantic Kernel
def setup_kernel():
    # Load Azure OpenAI credentials from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([endpoint, api_key, deployment_name]):
        raise ValueError("Azure OpenAI environment variables are not properly set.")

    # Initialize the Kernel
    kernel = Kernel()

    # Add Azure OpenAI chat model
    kernel.add_chat_completion_service(
        deployment_name,
        AzureChatCompletion(
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
        ),
    )

    return kernel

# Truncate chat history based on the number of conversations
def truncate_chat_history_by_conversations(chat_history, max_conversations):
    """
    Truncate chat history to retain only the last `max_conversations` conversations.
    A conversation is a pair of user input and assistant response.
    """
    num_messages = len(chat_history.messages)
    num_conversations = num_messages // 2

    if num_conversations > max_conversations:
        # Calculate the number of messages to keep (each conversation is 2 messages)
        messages_to_keep = max_conversations * 2
        chat_history.messages = chat_history.messages[-messages_to_keep:]

# Initialize the Kernel and Chat History
kernel = setup_kernel()
chat_history = ChatHistory()

# Define the maximum number of conversations to retain in memory
max_conversations = 3

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Get the user input from the request
        user_input = request.json.get("input")

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Add the user input to the chat history
        chat_history.add_user_message(user_input)

        # Get the assistant's response
        response = kernel.chat_completion.complete(chat_history)

        # Add the assistant's response to the chat history
        chat_history.add_assistant_message(response)

        # Truncate the chat history based on the number of conversations
        truncate_chat_history_by_conversations(chat_history, max_conversations)

        # Return the assistant's response
        return jsonify({"response": response}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
