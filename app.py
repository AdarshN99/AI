# app.py
import streamlit as st
import os
import requests
import json
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import RawVectorQuery
from dotenv import load_dotenv

load_dotenv()

# Azure Search setup
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# Retrieve vector for the uploaded image
def get_image_vector(image):
    headers = {'Ocp-Apim-Subscription-Key': os.getenv("AZURE_AI_VISION_API_KEY")}
    response = requests.post(
        f"{os.getenv('AZURE_AI_VISION_ENDPOINT')}/computervision/retrieval:vectorizeImage?api-version=2023-04-01-preview",
        headers=headers, files={'image': image})
    return response.json().get("vector")

# UI setup
st.title("Image Similarity Search")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    vector = get_image_vector(uploaded_image)
    
    # Perform vector search
    vector_query = RawVectorQuery(vector=vector, k=3, fields="image_vector")
    results = search_client.search(search_text=None, vector_queries=[vector_query], select=["description"])

    st.subheader("Search Results")
    for result in results:
        st.write(result["description"])
        image_path = os.path.join("images", result["description"])
        if os.path.exists(image_path):
            st.image(image_path, caption=result["description"], use_column_width=True)
