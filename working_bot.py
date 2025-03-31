import os
import groq
import feedparser
from pinecone import Pinecone,ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"]="pcsk_29rn7z_TgJfTP7rwj4xPuzBm5brBJcomX68uZjXZErNLhetYjrakwZUp2eaQCo59TzjJmq"
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.delete_index("ragproj")
index_name = "ragproj"

if index_name not in pc.list_indexes():
    pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
    ) 
index = pc.Index(index_name)

# ✅ Initialize Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Fetch & Store RSS Articles in Pinecone
def fetch_rss():
    urls = [
        "https://rss.app/feeds/tQh5rZ0KkQjQzkBZ.xml",
        "https://rss.app/feeds/tENrTJiUNVEhS8tW.xml",
        "https://rss.app/feeds/tQEw8VZnrxt9SmlX.xml",
        "https://rss.app/feeds/tzGepaHeQJulAlTq.xml"
    ]
    starred_articles = []
    
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article = {
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get("summary", "No summary available")
            }
            starred_articles.append(article)

            # ✅ Convert text to embedding & store in Pinecone
            vector = model.encode(article["summary"]).tolist()
            index.upsert([(article["link"], vector, {"title": article["title"], "summary": article["summary"]})])
    
    print(f"✅ {len(starred_articles)} articles stored in Pinecone!")

# ✅ Retrieve Relevant Data from Pinecone for RAG
def retrieve_articles(query, top_k=3):
    query_vector = model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    retrieved_text = ""
    for match in result["matches"]:
        retrieved_text += f"\n🔗 {match['metadata']['title']} - {match['metadata']['summary']}"
    
    return retrieved_text

# ✅ Process PDF File & Extract Text
def process_pdf():
    input_pdf = input("Enter absolute path to your PDF (Example: C:\\Users\\XYZ\\file.pdf): ").strip()
    if not os.path.exists(input_pdf):
        print("❌ Invalid file path.")
        return

    reader = PdfReader(input_pdf)
    extracted_text = ""

    print("\n🔄 Processing PDF...\n")

    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
        else:
            print("⚠️ Some pages may contain images instead of text.")

    if extracted_text:
        query_with_pdf(extracted_text)
    else:
        print("❌ No readable text found in the PDF.")

# ✅ Query Groq with RAG-based Context
def query_with_pdf(pdf_text):
    client = groq.Groq(api_key=groq_api_key)
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Analyze the following data without adding assumptions."},
            {"role": "user", "content": pdf_text}
        ],
        top_p=0.9,
        temperature=0.9,
        max_tokens=1000,
        stream=False
    )
    print(res.choices[0].message.content)

# ✅ Chatbot with Real-Time Data
def chat_with_rag(user_query):
    context = retrieve_articles(user_query)
    prompt = f"""
    You are an AI assistant with real-time news. Answer based on the latest updates:
    
    Question: {user_query}
    
    Latest News:
    {context}
    
    Answer:
    """

    client = groq.Groq(api_key=groq_api_key)
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ✅ Main Loop
print("Type 'rss' to update news, 'pdf' to process a document, or 'exit' to quit.")

while True:
    user_input = input("💬 You: ").strip().lower()
    
    if user_input == "rss":
        fetch_rss()
    elif user_input == "pdf":
        process_pdf()
    elif user_input == "exit":
        print("👋 Goodbye!")
        break
    else:
        print("🤖 AI:", chat_with_rag(user_input))
