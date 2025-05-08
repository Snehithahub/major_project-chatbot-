from flask import Flask, render_template_string, request, jsonify, session
import hashlib
import os
import feedparser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import groq
from pypdf import PdfReader
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline


os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"]="pcsk_4LvUu_UDLniiodx4vXGPxR4C7jp3kD6MS7KUm4CRMdmAZXa1UpVb28k7pv4xGHEzY4cQb"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Pinecone setup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer("all-MiniLM-L6-v2")
index_name = "ragproj"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(name=index_name)

# Hash function for URLs
def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

# Function to fetch RSS feed data
def fetch_rss():
    urls = [
        "https://rss.app/feeds/tQh5rZ0KkQjQzkBZ.xml",
        "https://rss.app/feeds/tENrTJiUNVEhS8tW.xml",
        "https://rss.app/feeds/tQEw8VZnrxt9SmlX.xml",
        "https://rss.app/feeds/tzGepaHeQJulAlTq.xml",
        "https://feeds.feedburner.com/ndtvnews-latest"
    ]

    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article_id = hash_url(entry.link)
            summary = entry.get("summary", "No summary available")
            title = entry.title
            link = entry.link
            vector = model.encode(summary).tolist()

            existing = index.query(vector=vector, top_k=1, include_metadata=True)
            if existing.get("matches") and existing["matches"][0]["id"] == article_id:
                continue  # Skip if already exists
            index.upsert([(article_id, vector, {"title": title, "summary": summary, "link": link})])

# Retrieve articles based on query
def retrieve_articles(query, top_k=3):
    query_vector = model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    if not result or "matches" not in result:
        return "No relevant articles found."

    return "\n".join(
        [f"{m['metadata']['title']} - {m['metadata']['summary']}" for m in result.get("matches", [])]
    )

# Function to process PDF and extract text
def process_pdf(pdf_file):
    try:
        pdf = PdfReader(pdf_file)
        text = ""
        for page in pdf.pages[:3]:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Function to query PDF content using Groq
def query_with_pdf(pdf_text):
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "Analyze the following data without assumptions."}, {"role": "user", "content": pdf_text}]
    )
    return res.choices[0].message.content

# Function to generate image caption from uploaded image
def generate_prompt(input_image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    image = Image.open(input_image).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    caption_ids = model.generate(**inputs)
    prompt = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return prompt

# Function to generate similar image using Stable Diffusion
def generate_similar_image(text_prompt, device="cuda" if torch.cuda.is_available() else "cpu"):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    image = pipe(text_prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path

# Home route to serve the main page
@app.route("/", methods=["GET", "POST"])
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []

    return render_template_string(HTML_TEMPLATE, chat_history=session['chat_history'])

# Chat route to handle message exchanges
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("user_message", "").strip()
    file = request.files.get("file")
    file_type = request.form.get("type")
    prompt = ""

    if file:
        if file_type == "pdf":
            pdf_text = process_pdf(file)
            prompt = f"Summarize this PDF content:\n{pdf_text[:2000]}"
        elif file_type == "image":
            image_prompt = generate_prompt(file)
            prompt = f"Describe this image: {image_prompt}"
        else:
            return jsonify({"response": "❌ Unsupported file type."})
    elif user_message:
        prompt = f"User: {user_message}\nAssistant:"

    # Send the prompt to Groq for response
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    session.setdefault('chat_history', []).append({'sender': 'user', 'message': user_message or f"{file_type.upper()} uploaded"})
    session['chat_history'].append({'sender': 'bot', 'message': answer})

    return jsonify({"response": answer})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Neon Chatbot</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    .header {
      padding: 20px;
      font-size: 24px;
      text-align: center;
      color: #00ffff;
      background: #121212;
      box-shadow: 0 0 10px #00ffff44;
    }
    .chat-history {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background: #1a1a1a;
      display: flex;
      flex-direction: column;
      gap: 10px;
      scrollbar-width: thin;
    }
    .chat-bubble {
      padding: 12px 16px;
      max-width: 70%;
      border-radius: 15px;
      margin: 4px 0;
      animation: fadeIn 0.3s ease;
    }
    .user-bubble {
      background-color: #007bff33;
      align-self: flex-end;
      color: #00ffff;
    }
    .bot-bubble {
      background-color: #2a2a2a;
      align-self: flex-start;
      color: #ffffff;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .chat-controls {
      display: flex;
      flex-direction: column;
      padding: 10px;
      background: #121212;
    }
    .input-row {
      display: flex;
      gap: 10px;
    }
    .chat-controls input[type="text"] {
      flex: 1;
      padding: 10px;
      background: #202020;
      border: 1px solid #333;
      border-radius: 10px;
      color: #fff;
    }
    .chat-controls button {
      padding: 10px 15px;
      background-color: #00ffff44;
      border: 1px solid #00ffffaa;
      color: #00ffff;
      border-radius: 10px;
      cursor: pointer;
      transition: 0.2s;
    }
    .chat-controls button:hover {
      background-color: #00ffff88;
    }
    .file-row {
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }
    .file-row button {
      flex: 1;
    }
    input[type="file"] {
      display: none;
    }
  </style>
</head>
<body>
  <div class="header">🎮 AI Companion</div>

  <div id="chat-history" class="chat-history"></div>

  <div class="chat-controls">
    <div class="input-row">
      <input type="text" id="user_message" placeholder="Type a message..." />
      <button onclick="sendMessage()">Send</button>
    </div>
    <div class="file-row">
      <button onclick="document.getElementById('pdf_input').click()">📄 Upload PDF</button>
      <button onclick="document.getElementById('image_input').click()">🖼️ Upload Image</button>
      <button onclick="fetchRSS()">🔄 Refresh RSS</button>
    </div>
    <input type="file" id="pdf_input" accept=".pdf" onchange="uploadFile(this.files[0], 'pdf')" />
    <input type="file" id="image_input" accept="image/*" onchange="uploadFile(this.files[0], 'image')" />
  </div>

  <script>
    const chatHistory = document.getElementById("chat-history");

    function appendMessage(message, sender = "bot") {
      const div = document.createElement("div");
      div.className = `chat-bubble ${sender}-bubble`;
      div.textContent = message;
      chatHistory.appendChild(div);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function sendMessage() {
      const message = document.getElementById("user_message").value.trim();
      if (!message) return;

      appendMessage(message, "user");

      const formData = new FormData();
      formData.append("user_message", message);

      fetch("/chat", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => appendMessage(data.response));
      
      document.getElementById("user_message").value = "";
    }

    function uploadFile(file, type) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("type", type);

      appendMessage(`${type.toUpperCase()} uploaded`, "user");

      fetch("/chat", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => appendMessage(data.response));
    }

    function fetchRSS() {
      sendMessageWithPrompt("Fetch latest RSS updates");
    }

    function sendMessageWithPrompt(prompt) {
      appendMessage(prompt, "user");

      const formData = new FormData();
      formData.append("user_message", prompt);

      fetch("/chat", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => appendMessage(data.response));
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
