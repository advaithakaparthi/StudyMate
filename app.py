from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from session_manager import create_session, get_documents_path, get_vector_store_path, save_chat_history, load_chat_history
from ingest import process_documents
from agent import get_or_create_agent
from langchain_core.messages import HumanMessage

app = Flask(__name__)

@app.route('/')
def index():
    sessions = os.listdir('sessions')
    return render_template('index.html', sessions=sessions)

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = create_session()
    documents_path = get_documents_path(session_id)
    files = request.files.getlist('documents')
    for file in files:
        file.save(os.path.join(documents_path, file.filename))
    process_documents(documents_path, get_vector_store_path(session_id))
    return redirect(url_for('chat', session_id=session_id))

@app.route('/chat/<session_id>')
def chat(session_id):
    chat_history = load_chat_history(session_id)
    return render_template('chat.html', session_id=session_id, chat_history=chat_history)

@app.route('/ask/<session_id>', methods=['POST'])
def ask(session_id):
    user_question = request.form['question']
    agent = get_or_create_agent(session_id)
    messages = [HumanMessage(content=user_question)]
    result = agent.invoke({"messages": messages, "session_id": session_id})
    response = result['messages'][-1].content

    chat_history = load_chat_history(session_id)
    chat_history.append(("You", user_question))
    chat_history.append(("DocuMentor", response))
    save_chat_history(session_id, chat_history)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
