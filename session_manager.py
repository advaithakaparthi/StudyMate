import os
import uuid
import json

BASE_DIR = 'sessions'
os.makedirs(BASE_DIR, exist_ok=True)

def create_session():
    session_id = str(uuid.uuid4())
    session_path = os.path.join(BASE_DIR, session_id)
    os.makedirs(os.path.join(session_path, 'documents'), exist_ok=True)
    os.makedirs(os.path.join(session_path, 'vector_store'), exist_ok=True)
    save_chat_history(session_id, [])
    return session_id

def get_session_path(session_id):
    return os.path.join(BASE_DIR, session_id)

def get_documents_path(session_id):
    return os.path.join(get_session_path(session_id), 'documents')

def get_vector_store_path(session_id):
    return os.path.join(get_session_path(session_id), 'vector_store')

def save_chat_history(session_id, history):
    with open(os.path.join(get_session_path(session_id), 'chat_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def load_chat_history(session_id):
    history_path = os.path.join(get_session_path(session_id), 'chat_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
