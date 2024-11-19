import re
import os
import json
import base64
import logging
import openai
import http.client, urllib.parse
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from tenacity import retry, stop_after_attempt, wait_fixed
from azure.core.exceptions import HttpResponseError, ServiceRequestError  
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
from azure_openai import *  
from config import *  

credential = AzureKeyCredential(searchkey)
text_search_client = SearchClient(endpoint=service_endpoint, index_name=index, credential=credential)
image_search_client = SearchClient(endpoint=service_endpoint, index_name=index_image, credential=credential)


app = Flask(__name__)
CORS(app, supports_credentials=True)


logging.basicConfig(level=logging.INFO)

@app.route('/devedgesearch', methods=['POST'])
@cross_origin(supports_credentials=True)
def text_search():
    try:
        user_input = request.json.get("question")
        if not user_input:
            raise ValueError("Prompt is required")

        # Define field mappings for Azure Search
        KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT", "content")
        KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY", "category")
        KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE", "sourcepage")
        exclude_category = None
        filter_condition = f"category ne '{exclude_category.replace("'", "''")}'" if exclude_category else None

        # Search query to Azure Search with error handling
        try:
            results = text_search_client.search(
                user_input,
                filter=filter_condition,
                query_type=QueryType.SEMANTIC,
                #query_language="en-us",
                #query_speller="lexicon",
                semantic_configuration_name="default",
                top=3
            )
        except HttpResponseError as e:
            logging.error(f"Azure Search API error: {e.message}")
            if e.status_code == 400:
                def error_stream():
                    yield f"status: 400 \ncode: BAD_REQUEST \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 401:
                def error_stream():
                    yield f"status: 401 \ncode: UNAUTHORIZED \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 403:
                def error_stream():
                    yield f"status: 403 \ncode: FORBIDDEN \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 404:
                def error_stream():
                    yield f"status: 404 \ncode: NOT_FOUND \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 409:
                def error_stream():
                    yield f"status: 409 \ncode: CONFLICT \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 422:
                def error_stream():
                    yield f"status: 422 \ncode: UNPROCESSABLE_ENTITY \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 500:
                def error_stream():
                    yield f"status: 500 \ncode: INTERNAL_SERVER_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            elif e.status_code == 503:
                def error_stream():
                    yield f"status: 503 \ncode: SERVICE_UNAVAILABLE \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            else:
                def error_stream():
                    yield f"status: {e.status_code} \ncode: AZURE_SEARCH_API_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

        except ServiceRequestError as e:
            logging.error(f"Service request error: {e}")
            def error_stream():
                yield f"status: 504 \ncode: SERVICE_REQUEST_ERROR \nerror: {str(e)}\n"
            return Response(stream_with_context(error_stream()), content_type="event-stream")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            def error_stream():
                yield f"status: 500 \ncode: INTERNAL_ERROR \nerror: {str(e)}\n"
            return Response(stream_with_context(error_stream()), content_type="event-stream")

        content = "\n".join([f"{doc[KB_FIELDS_SOURCEPAGE]}: {doc[KB_FIELDS_CONTENT].replace('\n', '').replace('\r', '')}" for doc in results])
    

        def generate_response():
            try:
                conversation = [{"role": "system", "content": "Assistant is a great language model formed by OpenAI."}]
                prompt = create_prompt(content, user_input)
                conversation.append({"role": "assistant", "content": prompt})
                conversation.append({"role": "user", "content": user_input})
                reply = client.chat.completions.create(
                    model=deployment_id_gpt4,
                    messages=conversation,
                    temperature=0,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=True,
                    stop = [' END']
                    )
                
                for chunk in reply:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            except ValueError as e:
                logging.error(f"Value error: {e}")
                def error_stream():
                    yield f"status: 400 \ncode: INVALID_ARGUMENT \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.APIConnectionError as e:
                logging.error(f"API connection error: {e}")
                def error_stream():
                    yield f"status: 503 \ncode: API_CONNECTION_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.APITimeoutError as e:
                logging.error(f"API timeout error: {e}")
                def error_stream():
                    yield f"status: 504 \ncode: TIMEOUT \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.AuthenticationError as e:
                logging.error(f"Authentication error: {e}")
                def error_stream():
                    yield f"status: 401 \ncode: UNAUTHENTICATED \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.BadRequestError as e:
                logging.error(f"Bad request error: {e}")
                def error_stream():
                    yield f"status: 400 \ncode: BAD_REQUEST_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.ConflictError as e:
                logging.error(f"Conflict error: {e}")
                def error_stream():
                    yield f"status: 409 \ncode: CONFLICT \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.InternalServerError as e:
                logging.error(f"Internal server error: {e}")
                def error_stream():
                    yield f"status: 500 \ncode: INTERNAL_SERVER_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.NotFoundError as e:
                logging.error(f"Not found error: {e}")
                def error_stream():
                    yield f"status: 404 \ncode: NOT_FOUND \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.PermissionDeniedError as e:
                logging.error(f"Permission denied error: {e}")
                def error_stream():
                    yield f"status: 403 \ncode: PERMISSION_DENIED \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.RateLimitError as e:
                logging.error(f"Rate limit exceeded: {e}")
                def error_stream():
                    yield f"status: 429 \ncode: QUOTA_EXCEEDED \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except openai.UnprocessableEntityError as e:
                logging.error(f"Unprocessable entity error: {e}")
                def error_stream():
                    yield f"status: 422 \ncode: UNIDENTIFIABLE_DEVICE \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")

            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                def error_stream():
                    yield f"status: 500 \ncode: INTERNAL_ERROR \nerror: {str(e)}\n"
                return Response(stream_with_context(error_stream()), content_type="event-stream")
                    
            return Response(stream_with_context(generate_response()),content_type="event-stream")  

    except ValueError as e:
        logging.error(f"Value error: {e}")
        def error_stream():
            yield f"Error: {str(e)}\n"
        return Response(stream_with_context(error_stream()),content_type="event-stream")
            
    except Exception as e:
            logging.error(f"Unexpected error: {e}")
            def error_stream():
                yield f"Error: Unexpected error - {str(e)}\n"
            return Response(stream_with_context(error_stream()),content_type="event-stream")


def sanitize_id(filename):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)


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

        if response.status == 200:
            return data.get("vector")
        elif response.status == 401:
            raise PermissionError("Unauthorized: Check API Key")
        elif response.status == 400:
            raise ValueError("Bad Request")
        elif response.status == 429:
            raise ConnectionError("Too many requests")
        elif response.status == 503:
            raise PermissionError("Service Unavailable")
        else:
            raise Exception(f"Unexpected Error:{response.status} {response.reason}")
       
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found {image_path}")
    except http.client.HTTPException as e:
        raise ConnectionError(f"HTTP Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/upload_images', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_all_images():
    files = os.listdir(FILE_PATH_IMG)
    image_embeddings = {}
    documents = []

    try:
        for file in files:
            file_path = os.path.join(FILE_PATH_IMG, file)
            image_embeddings[file] = get_image_vector(file_path, aiVisionApiKey, aiVisionRegion)
        
        for counter, file in enumerate(files):
            sanitized_id = sanitize_id(file)
            document = {
                "id": sanitized_id,
                "description": file,
                "image_vector": image_embeddings[file]
            }
            documents.append(document)
    except Exception as e:
         return jsonify({ "Error processing files": str(e)})
    

    if documents:
        result = image_search_client.upload_documents(documents)
        return jsonify({"message":f"Uploaded {len(documents)} documents successfully."})
    else:
        return jsonify("No files to upload")

@app.route('/imagesearch', methods=['POST'])
@cross_origin(supports_credentials=True)
def image_search():
    try:
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
            k_nearest_neighbors=2,  
            fields="image_vector", 
        )
        try:
        # Perform the search using VectorizedQuery
            results = image_search_client.search(
                search_text=None, 
                vector_queries=[vectorized_query],
                select=["description"]
            )
        except HttpResponseError as e:
            if e.status_code == 400:
                return jsonify({"status": 400, "code": "BAD_REQUEST", "message": e.message}), 400
            elif e.status_code == 401:
                return jsonify({"status": 401, "code": "UNAUTHORIZED", "message": e.message}), 401
            elif e.status_code == 403:
                return jsonify({"status": 403, "code": "FORBIDDEN", "message": e.message}), 403
            elif e.status_code == 404:
                return jsonify({"status": 404, "code": "NOT_FOUND", "message": e.message}), 404
            elif e.status_code == 409:
                return jsonify({"status": 409, "code": "CONFLICT", "message": e.message}), 409
            elif e.status_code == 422:
                return jsonify({"status": 422, "code": "UNPROCESSABLE_ENTITY", "message": e.message}), 422
            elif e.status_code == 500:
                return jsonify({"status": 500, "code": "INTERNAL_SERVER_ERROR", "message": e.message}), 500
            elif e.status_code == 503:
                return jsonify({"status": 503, "code": "SERVICE_UNAVAILABLE", "message": e.message}), 503
            else:
                return jsonify({"status": e.status_code, "code": "UNEXPECTED_ERROR", "message": e.message}), e.status_code
        except ServiceRequestError as e:
            return jsonify({"status": 504, "code": "SERVICE_REQUEST_ERROR", "message": str(e)}), 504
        except Exception as e:
            return jsonify({"status": 500, "code": "INTERNAL_ERROR", "message": str(e)}), 500


        similar_images = []
        for result in results:
            image_path = os.path.join(FILE_PATH_IMG, result["description"])
            encoded_image = image_to_base64(image_path) 
            similar_images.append({
                "image_name": result["description"],
                "image_base64": encoded_image
            })

        return jsonify({"similar_images": similar_images})
    
    except TimeoutError as e:
        return jsonify({ "status": 504 ,"code": "INVALID_ARGUMENT","message": str(e)}), 504
    except Exception as e:
        return jsonify({ "status": 500 ,"code": "INTERNAL","message": str(e)}), 500

if __name__ == '__main__':
    app.run()
