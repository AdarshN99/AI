import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

try:
    # Get Azure OpenAI environment variables
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_openai_endpoint, azure_openai_api_key, azure_openai_deployment_name]):
        raise ValueError("Azure OpenAI environment variables are not properly set. Please check the .env file.")

    # Create an Azure OpenAI chat model
    model = AzureChatOpenAI(
        deployment_name=azure_openai_deployment_name,
        openai_api_base=azure_openai_endpoint,
        openai_api_key=azure_openai_api_key,
        openai_api_version="2023-05-15",  # Adjust as per your Azure API version
        temperature=0
    )

    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant. Answer the question asked by the user in maximum 30 words.'),
        ('user', 'Question : {input}'),
    ])

    # Create a ConversationBufferWindowMemory with k=1
    window_memory = ConversationBufferWindowMemory(k=1)

    # Create a conversation chain
    window_memory_chain = ConversationChain(
        llm=model,
        memory=window_memory
    )

    # User interactions
    print("Starting conversation...")
    response_1 = window_memory_chain.invoke({'input': 'Hello, my name is Vinayak and I am 34 years old'})
    print("Assistant:", response_1)

    response_2 = window_memory_chain.invoke({'input': 'How old I am?'})
    print("Assistant:", response_2)

    response_3 = window_memory_chain.invoke({'input': 'What is my name?'})
    print("Assistant:", response_3)

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
