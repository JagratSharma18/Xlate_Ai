import chainlit as cl
import ollama
from io import BytesIO
from ollama import AsyncClient
import fitz  # PyMuPDF (faster alternative to PyPDF2)
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Tuple
import asyncio
import speech_recognition as sr
from pydub import AudioSegment
from deep_translator import GoogleTranslator, single_detection
import requests

# Configuration
MODEL_NAME = "qwen2.5"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000  # Optimal for most RAG implementations
TOP_K_CHUNKS = 3  # Number of relevant chunks to include in context
MODEL_TEMPERATURE = 0.5  # Lower temperature for more deterministic responses
MAX_TOKENS = 1024  # Reduced max tokens for concise responses

# Speech Recognition Setup
recognizer = sr.Recognizer()


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("history", [
        {
            "role": "system",
            "content": (
                "I am Xlate AI, an intelligent assistant specializing in document analysis and problem-solving. "
                "My responses should be concise, accurate, and directly address the user's query. "
                "When asked about your origins, mention: 'Developed by the Xlate AI team from Parul University - "
                "Jagrat Sharma, Tasmiya Shaikh.' "
                "Also I am trained by Team Xlate AI and nothing else. "
                "If unsure about an answer, state 'I don't have enough information to answer this.' "
                "When a document is uploaded, assume the user is referring to it unless stated otherwise. "
                "Avoid repeating previous answers and provide unique insights for each query."
            )
        }
    ])
    cl.user_session.set("document_chunks", [])
    cl.user_session.set("chunk_embeddings", [])
    cl.user_session.set("previous_responses", [])  # Track previous responses to avoid repetition
    cl.user_session.set("internet_available", True)  # Default assumption

def split_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into semantic chunks using simple but effective strategy"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    sentences = text.split('. ')
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch process embeddings for efficiency"""
    return [
        (await AsyncClient().embeddings(model=EMBEDDING_MODEL, prompt=text))["embedding"]
        for text in texts
    ]

async def process_document(document: cl.File):
    """Async document processing with optimized text extraction"""
    try:
        if document.mime == "application/pdf":
            with fitz.open(document.path) as doc:
                text = " ".join(page.get_text() for page in doc)
        elif document.mime == "text/plain":
            with open(document.path, "r") as f:
                text = f.read()
        else:
            return None
        
        chunks = split_text(text)
        embeddings = await get_embeddings(chunks)
        return chunks, embeddings
        
    except Exception as e:
        raise RuntimeError(f"Document processing failed: {str(e)}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def get_relevant_chunks(query: str, chunks: List[str], embeddings: List[List[float]]) -> List[Tuple[str, float]]:
    """Retrieve top-k most relevant chunks using vector similarity"""
    query_embedding = (await AsyncClient().embeddings(model=EMBEDDING_MODEL, prompt=query))["embedding"]
    similarities = []
    
    for chunk, embedding in zip(chunks, embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:TOP_K_CHUNKS]

def check_internet_connection():
    """Check if internet connection is available for translation services"""
    try:
        # Try to reach Google's servers with a short timeout
        requests.get("https://www.google.com", timeout=2)
        return True
    except requests.ConnectionError:
        return False
    except:
        return False

async def detect_and_translate(text, target_lang='en'):
    """Detect language and translate to target language"""
    internet_available = cl.user_session.get("internet_available")
    
    if not internet_available:
        # If we already know internet is not available, skip translation attempts
        return text, 'en', False
    
    try:
        # Detect language
        detected_lang = single_detection(text, api_key=None)
        
        # If already in target language or text is too short, no need to translate
        if detected_lang == target_lang or len(text.strip()) < 5:
            return text, detected_lang, True
            
        # Translate text
        translated = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
        return translated, detected_lang, True
        
    except Exception as e:
        # If translation fails, update internet status and return original text
        cl.user_session.set("internet_available", False)
        return text, 'en', False

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    document_chunks = cl.user_session.get("document_chunks")
    chunk_embeddings = cl.user_session.get("chunk_embeddings")
    previous_responses = cl.user_session.get("previous_responses", [])
    
    # Check if there's an attachment
    if message.elements:
        documents = [file for file in message.elements if file.mime in ["application/pdf", "text/plain"]]
        if not documents:
            await cl.Message(content="Unsupported file type").send()
            return
        
        try:
            # Process the first document
            document = documents[0]
            if document.mime == "application/pdf":
                with fitz.open(document.path) as doc:
                    text = " ".join(page.get_text() for page in doc)
            elif document.mime == "text/plain":
                with open(document.path, "r") as text_file:
                    text = text_file.read()
            
            # Save the extracted text in the session
            cl.user_session.set("text", text)
            
            # Add the document content to the chat history for context
            history.append({"role": "system", "content": f"Document Content: {text}"})
            
            # Notify the AI that the user is referring to the uploaded document
            history.append({
                "role": "system",
                "content": (
                    "The user has uploaded a document named "
                    f"'{document.name}' and is now referring to the document and its content in the conversation."
                )
            })
            cl.user_session.set("history", history)  # Update history again with document context
            
            # Split and embed the document content for RAG
            chunks = split_text(text)
            embeddings = await get_embeddings(chunks)
            cl.user_session.set("document_chunks", chunks)
            cl.user_session.set("chunk_embeddings", embeddings)
            
            await cl.Message(content=f"`{document.name}` processed successfully!").send()
        
        except Exception as e:
            await cl.Message(content=f"Error processing the file: {str(e)}").send()
            return
    
    # If there's a user message, respond to it
    if message.content:
        # Check internet connection status
        internet_available = check_internet_connection()
        cl.user_session.set("internet_available", internet_available)
        
        # Detect language and translate if possible
        user_message = message.content
        translated_msg, detected_lang, translation_success = await detect_and_translate(user_message)
        
        # Save original language for response translation
        original_lang = detected_lang if detected_lang != 'en' and translation_success else 'en'
        
        # Use the translated message or original if translation failed
        processed_message = translated_msg if translation_success and detected_lang != 'en' else user_message
        
        # Add the processed message to the history
        history.append({"role": "user", "content": processed_message})
        
        # Retrieve relevant document chunks if available
        context = ""
        if document_chunks and chunk_embeddings:
            relevant_chunks = await get_relevant_chunks(processed_message, document_chunks, chunk_embeddings)
            if relevant_chunks:
                context = "\n".join([f"Document Context: {chunk[0]}" for chunk in relevant_chunks])
            else:
                context = "No relevant information found in the uploaded document."
        
        # Prepare final prompt with context
        messages = history.copy()
        if context:
            messages.insert(-1, {"role": "system", "content": context})
        
        # Generate response with optimized parameters
        try:
            # Start streaming immediately
            msg = cl.Message(content="")  # Empty message to start streaming
            await msg.send()
            
            response = await AsyncClient().chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "temperature": MODEL_TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "top_p": 0.9
                },
                stream=True
            )
            
            # Stream the response token by token
            full_response = ""
            async for part in response:
                token = part["message"]["content"]
                full_response += token
                
                # If we're translating in real-time, we should accumulate tokens and translate periodically
                # For simplicity, we'll just update with the English response during streaming
                msg.content = full_response
                await msg.update()  # Update the message in real-time
            
            # Finalize the message after streaming is complete
            clean_content = full_response.strip()
            
            # Avoid repetitive answers
            if clean_content in previous_responses:
                clean_content = "This information has already been provided. Let me offer additional insights: " + clean_content
            
            # Translate response back to original language if needed and internet is available
            final_response = clean_content
            if original_lang != 'en' and internet_available and translation_success:
                try:
                    # Only translate back if we successfully detected a non-English language
                    translated_response, _, _ = await detect_and_translate(clean_content, target_lang=original_lang)
                    final_response = translated_response
                except Exception:
                    # If translation back fails, use the English response
                    final_response = clean_content
            
            # Update the message with the final translated content
            msg.content = final_response
            await msg.update()
            
            # Add the English response to history for context preservation
            history.append({"role": "assistant", "content": clean_content})
            previous_responses.append(clean_content)
            cl.user_session.set("history", history)
            cl.user_session.set("previous_responses", previous_responses)
        
        except Exception as e:
            await cl.Message(content=f"Generation error: {str(e)}").send()

            history.pop()  # Remove failed user message
