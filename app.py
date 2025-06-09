import os
import re
import json
import requests
import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# === ENVIRONMENT VARIABLES FROM HF SECRETS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PLAY_HT_API_KEY = os.getenv("PLAY_HT_API_KEY")
PLAY_HT_USER_ID = os.getenv("PLAY_HT_USER_ID")
PLAY_HT_VOICE_ID = "s3://voice-cloning-zero-shot/a5201a3e-a9c5-4d86-b95c-6300dfd5100a/original/manifest.json"  

play_ht_api_get_audio_url = "https://play.ht/api/v2/tts"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === GROQ + LANGCHAIN SETUP ===
llm = ChatGroq(
    temperature=0.5,
    model_name="llama-3.3-70b-versatile" 
)

template = """You are a helpful assistant to answer user queries.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"],
    template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# === PLAYHT AUDIO GENERATION ===
headers = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    "AUTHORIZATION": "Bearer " + PLAY_HT_API_KEY,
    "X-USER-ID": PLAY_HT_USER_ID
}

def get_payload(text):
    return {
        "text": text,
        "voice": PLAY_HT_VOICE_ID,
        "quality": "medium",
        "output_format": "mp3",
        "speed": 1,
        "sample_rate": 24000,
        "seed": None,
        "temperature": None
    }

def get_generated_audio(text):
    payload = get_payload(text)
    generated_response = {}
    try:
        response = requests.post(play_ht_api_get_audio_url, json=payload, headers=headers)
        response.raise_for_status()
        generated_response["type"] = 'SUCCESS'
        generated_response["response"] = response.text
    except requests.exceptions.RequestException:
        generated_response["type"] = 'ERROR'
        try:
            response_text = json.loads(response.text)
            generated_response["response"] = response_text.get('error_message', response.text)
        except:
            generated_response["response"] = response.text
    return generated_response

def extract_urls(text):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*'
    return re.findall(url_pattern, text)

def get_audio_reply_for_question(text):
    generated_audio_event = get_generated_audio(text)
    final_response = {"audio_url": '', "message": ''}
    if generated_audio_event["type"] == 'SUCCESS':
        audio_urls = extract_urls(generated_audio_event["response"])
        if audio_urls:
            final_response['audio_url'] = audio_urls[-1]
        else:
            final_response['message'] = "No audio URL found"
    else:
        final_response['message'] = generated_audio_event['response']
    return final_response

def download_url(url):
    final_response = {'content': '', 'error': ''}
    try:
        response = requests.get(url)
        if response.status_code == 200:
            final_response['content'] = response.content
        else:
            final_response['error'] = f"Download failed: {response.status_code}"
    except Exception as e:
        final_response['error'] = f"Download error: {e}"
    return final_response

def get_filename_from_url(url):
    return os.path.basename(url)

def get_text_response(user_message):
    return llm_chain.predict(user_message=user_message)

def get_text_response_and_audio_response(user_message):
    response = get_text_response(user_message)
    audio_response = get_audio_reply_for_question(response)
    final_response = {'output_file_path': '', 'message': ''}
    audio_url = audio_response['audio_url']
    if audio_url:
        filename = get_filename_from_url(audio_url)
        download = download_url(audio_url)
        if download['content']:
            with open(filename, "wb") as audio_file:
                audio_file.write(download['content'])
                final_response['output_file_path'] = filename
        else:
            final_response['message'] = download['error']
    else:
        final_response['message'] = audio_response['message']
    return final_response

# === GRADIO CHAT UI ===
def chat_bot_response(message, history):
    result = get_text_response_and_audio_response(message)
    if result['output_file_path']:
        return (result['output_file_path'],)
    return result['message']

demo = gr.ChatInterface(
    chat_bot_response,
    examples=[
        "How are you doing?",
        "Tell me a joke!",
        "What are your interests?",
        "Which places do you like to visit?"
    ]
)

demo.launch()
