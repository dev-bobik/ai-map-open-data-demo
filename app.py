from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from vector_store import VectorStore

app = Flask(__name__)

# Load embedding model lazily
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

vs = VectorStore()
vs.load()

# Lazy-loaded chatbot variables
chatbot = None
chatbot_lock = threading.Lock()
chatbot_training = False

def simple_fallback_response(message: str) -> str:
    """Very small fallback bot (no external libs) for instant replies."""
    text = message.lower()
    if any(g in text for g in ["ahoj", "čau", "nazdar", "hola", "hello"]):
        return "Ahoj! Jak vám mohu pomoci?"
    if "map" in text or "mapa" in text:
        return "Tahle aplikace zobrazí mapu. Můžete na ni kliknout a přibližovat." 
    if any(q in text for q in ["dík", "děkuji", "thanks"]):
        return "Rádo se stalo!"
    return "Promiňte, tomu teď nerozumím — můžete to zkusit jinak?"

def call_hf_model(message: str) -> str:
    """Call a Hugging Face Inference API model if configured via env vars.
    Expects HF_API_KEY and HF_MODEL to be set. Returns model output or None on failure."""
    hf_key = os.environ.get('HF_API_KEY')
    hf_model = os.environ.get('HF_MODEL')
    if not hf_key or not hf_model:
        return None
    headers = {
        'Authorization': f'Bearer {hf_key}',
        'Content-Type': 'application/json'
    }
    payload = {"inputs": message}
    try:
        url = f'https://api-inference.huggingface.co/models/{hf_model}'
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # HF models sometimes return text directly or list of dicts
            if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                return data[0]['generated_text']
            if isinstance(data, dict) and 'generated_text' in data:
                return data['generated_text']
            # fallback: if it's a list with 'score' etc., join text keys
            if isinstance(data, list):
                return ' '.join(str(item.get('generated_text', '')) for item in data)
            return str(data)
        else:
            print('HF model call failed', resp.status_code, resp.text)
            return None
    except Exception as e:
        print('HF inference request error:', e)
        return None

def train_chatbot_in_background():
    global chatbot, chatbot_training
    try:
        from chatterbot import ChatBot
        from chatterbot.trainers import ChatterBotCorpusTrainer
    except Exception as e:
        print("ChatterBot not available or failed to import:", e)
        chatbot_training = False
        return

    with chatbot_lock:
        if chatbot is not None:
            chatbot_training = False
            return
        chatbot = ChatBot('MapBot')
        trainer = ChatterBotCorpusTrainer(chatbot)
        chatbot_training = True
    try:
        # Try Czech then English, otherwise skip
        try:
            trainer.train("chatterbot.corpus.czech")
        except Exception:
            trainer.train("chatterbot.corpus.english")
    except Exception as e:
        print("Warning: ChatterBot corpus training skipped or failed:", e)
    finally:
        chatbot_training = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_bot_response():
    # bezpečné získání JSON těla
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'response': ''}), 400
    user_message = data.get('message', '')
    # If chatbot is available, use it; otherwise use simple fallback
    with chatbot_lock:
        local_bot = chatbot
        local_training = chatbot_training

    if local_bot is not None:
        try:
            bot_response = str(local_bot.get_response(user_message))
            return jsonify({'response': bot_response})
        except Exception as e:
            print("Error using chatbot:", e)
    # Attempt Hugging Face model if configured
    hf_resp = call_hf_model(user_message)
    if hf_resp:
        return jsonify({'response': hf_resp})

    # If still training or chatbot not available, return fallback immediately
    return jsonify({'response': simple_fallback_response(user_message)})


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(silent=True)
    if not data or 'q' not in data:
        return jsonify({'error': 'missing q'}), 400
    q = data['q']
    model = get_embedding_model()
    q_emb = model.encode(q, convert_to_numpy=True).astype('float32')
    res = vs.search(q_emb, k=5)
    # assemble answer as concatenation of top texts (simple extractive)
    texts = [r['metadata']['text'] for row in res for r in row]
    answer = '\n\n'.join(texts[:3]) if texts else ''
    sources = [r['metadata'] for row in res for r in row]
    # If HF configured, attempt to generate a nicer answer using HF model
    hf_candidate = call_hf_model(q + '\n\n' + answer) if answer else None
    final = hf_candidate if hf_candidate else (answer if answer else simple_fallback_response(q))
    return jsonify({'answer': final, 'sources': sources})


if __name__ == '__main__':
    # Start chatbot training in background so server starts fast
    t = threading.Thread(target=train_chatbot_in_background, daemon=True)
    t.start()
    # small delay to let background thread start
    time.sleep(0.2)
    app.run(debug=True)