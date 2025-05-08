from flask import Flask, request, render_template_string, jsonify, session
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import groq
from pypdf import PdfReader
from PIL import Image
import io

# API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_4LvUu_UDLniiodx4vXGPxR4C7jp3kD6MS7KUm4CRMdmAZXa1UpVb28k7pv4xGHEzY4cQb"

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer("all-MiniLM-L6-v2")
index_name = "ragproj"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(name=index_name)

# HTML Template
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f7f7f7; margin: 0; padding: 0; }
    .chat-container { width: 100%; height: 100vh; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; padding: 20px; box-sizing: border-box; }
    .chat-history { width: 100%; max-width: 600px; max-height: 70%; overflow-y: auto; background: #fff; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); padding: 10px; margin-bottom: 20px; display: flex; flex-direction: column; gap: 15px; }
    .chat-bubble { padding: 10px; border-radius: 20px; max-width: 80%; margin: 5px 0; transition: 0.3s ease; }
    .user-bubble { background-color: #d1f1ff; align-self: flex-end; animation: userMessage 0.5s ease; }
    .bot-bubble { background-color: #f0f0f0; align-self: flex-start; animation: botMessage 0.5s ease; }
    @keyframes userMessage { from {opacity: 0; transform: translateX(30px);} to {opacity: 1; transform: translateX(0);} }
    @keyframes botMessage { from {opacity: 0; transform: translateX(-30px);} to {opacity: 1; transform: translateX(0);} }
    .chat-input { display: flex; width: 100%; max-width: 600px; padding: 10px; background-color: #fff; border-radius: 30px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); gap: 10px; }
    .chat-input input { width: 80%; padding: 10px; border-radius: 25px; border: 1px solid #ccc; outline: none; font-size: 16px; }
    .chat-input button { width: 15%; padding: 10px; background-color: #007bff; border: none; border-radius: 25px; color: white; font-size: 16px; cursor: pointer; transition: background-color 0.3s ease; }
    .chat-input button:hover { background-color: #0056b3; }
    .file-inputs { margin-top: 10px; display: flex; gap: 10px; justify-content: center; }
    .file-inputs label { cursor: pointer; color: #007bff; text-decoration: underline; }
  </style>
</head>
<body>
  <div class="chat-container">
    <div id="chat-history" class="chat-history">
      {% for msg in chat_history %}
        <div class="chat-bubble {{ msg['sender'] }}-bubble">{{ msg['message'] }}</div>
      {% endfor %}
    </div>

    <div class="chat-input">
      <input type="text" id="user-message" placeholder="Type your message..." />
      <button onclick="sendMessage()">Send</button>
    </div>

    <div class="file-inputs">
      <label>
        📄 Upload PDF
        <input type="file" id="pdf-upload" accept="application/pdf" style="display:none" onchange="uploadFile('pdf')" />
      </label>
      <label>
        🖼️ Upload Image
        <input type="file" id="image-upload" accept="image/*" style="display:none" onchange="uploadFile('image')" />
      </label>
    </div>
  </div>

  <script>
    function appendMessage(message, sender) {
      const chatHistory = document.getElementById('chat-history');
      const bubble = document.createElement('div');
      bubble.classList.add('chat-bubble', sender + '-bubble');
      bubble.textContent = message;
      chatHistory.appendChild(bubble);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function sendMessage() {
      const userMessage = document.getElementById('user-message').value.trim();
      if (!userMessage) return;
      appendMessage(userMessage, 'user');
      fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `user_message=${encodeURIComponent(userMessage)}`
      })
      .then(response => response.json())
      .then(data => appendMessage(data.response, 'bot'));
      document.getElementById('user-message').value = "";
    }

    function uploadFile(type) {
      const input = document.getElementById(type + '-upload');
      const file = input.files[0];
      if (!file) return;
      appendMessage(`📤 Uploading ${type.toUpperCase()}...`, 'user');
      const formData = new FormData();
      formData.append("file", file);
      formData.append("type", type);
      fetch('/chat', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => appendMessage(data.response, 'bot'))
      .catch(() => appendMessage("⚠️ File processing failed.", 'bot'));
    }
  </script>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []

    return render_template_string(HTML, chat_history=session['chat_history'])

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("user_message", "").strip()
    file = request.files.get("file")
    file_type = request.form.get("type")
    prompt = ""

    if file:
        if file_type == "pdf":
            try:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages[:3]:
                    text += page.extract_text() or ""
                prompt = f"Summarize this PDF content:\n{text[:2000]}"
            except:
                return jsonify({"response": "❌ Could not read the PDF."})
        elif file_type == "image":
            try:
                image = Image.open(file.stream).convert("RGB")
                prompt = "Describe this image."  # Placeholder, add BLIP if needed
            except:
                return jsonify({"response": "❌ Could not process the image."})
        else:
            return jsonify({"response": "❌ Unsupported file type."})
    elif user_message:
        prompt = f"User: {user_message}\nAssistant:"
    else:
        return jsonify({"response": "⚠️ No input received."})

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    session.setdefault('chat_history', []).append({'sender': 'user', 'message': user_message or f"{file_type.upper()} uploaded"})
    session['chat_history'].append({'sender': 'bot', 'message': answer})

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
