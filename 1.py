from flask import Flask, request, jsonify
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Initialize Flask app
app = Flask(__name__)

# Initialize Semantic Kernel
kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
service_id = "chat-gpt"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
    )
)

# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8

# Prompt template configuration
def setup_prompt_config(prompt):
    return PromptTemplateConfig(
        template=prompt,
        name="tldr",
        template_format="semantic-kernel",
        execution_settings=req_settings,
    )

# Add function to kernel
def add_function_to_kernel(prompt_template_config):
    return kernel.add_function(
        function_name="tldr_function",
        plugin_name="tldr_plugin",
        prompt_template_config=prompt_template_config,
    )

# API endpoint
@app.route('/generate-tldr', methods=['POST'])
def generate_tldr():
    try:
        # Extract the prompt from the request
        data = request.json
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Setup the prompt template configuration
        prompt_template_config = setup_prompt_config(prompt)

        # Add function to kernel
        function = add_function_to_kernel(prompt_template_config)

        # Async execution of the function
        async def invoke_kernel():
            return await kernel.invoke(function)

        # Run the async function
        result = asyncio.run(invoke_kernel())

        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
