from flask import Flask, render_template
import json

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)

# Vytvoření a trénování chatbota
chatbot = ChatBot('MapBot')
trainer = ChatterBotCorpusTrainer(chatbot)
try:
    # Nejprve se pokusíme natrénovat český korpus, pokud je dostupný
    trainer.train("chatterbot.corpus.czech")
except Exception as e:
    # Pokud český korpus není dostupný, pokusíme se spustit anglický korpus
    try:
        trainer.train("chatterbot.corpus.english")
    except Exception:
        # Pokud ani anglický korpus není dostupný, přeskočíme trénink
        print("Warning: ChatterBot corpus training skipped (no corpus available).", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    bot_response = str(chatbot.get_response(user_message))
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)