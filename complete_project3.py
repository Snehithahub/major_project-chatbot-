import os
import hashlib
import groq
import feedparser
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"]="pcsk_29rn7z_TgJfTP7rwj4xPuzBm5brBJcomX68uZjXZErNLhetYjrakwZUp2eaQCo59TzjJmq"
groq_api_key = os.getenv("GROQ_API_KEY")


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "ragproj"
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"🛠 Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(name=index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

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
                index.upsert([(article_id, vector, {"title": title, "summary": summary, "link": link})])
            else:
               
                index.upsert([(article_id, vector, {"title": title, "summary": summary, "link": link})])

    index_stats = index.describe_index_stats()
    total_vectors = index_stats['total_vector_count']
    print(f"RSS feed updated successfully!Total articles stored: {total_vectors}")

def retrieve_articles(query, top_k=3):
    query_vector = model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    if not result or "matches" not in result:
        return "No relevant articles found."
    
    retrieved_text = "\n".join(
        [f"{match['metadata']['title']} - {match['metadata']['summary']}" for match in result.get("matches", [])]
    )
    return retrieved_text

def process_pdf():
    input_pdf = input("Enter absolute path to your PDF (Example: C:\\Users\\XYZ\\file.pdf): ").strip()
    if not os.path.exists(input_pdf):
        print("Invalid file path.")
        return
    
    reader = PdfReader(input_pdf)
    extracted_texts = []
    
    print("\n Processing PDF...\n")
    
    for page in reader.pages[:5]:  
        text = page.extract_text()
        if text:
            extracted_texts.append(text)
        else:
            print("⚠️ Some pages may contain images instead of text.")
    
    extracted_text = "\n".join(extracted_texts)
    if extracted_text:
        query_with_pdf(extracted_text)
    else:
        print("❌ No readable text found in the PDF.")

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
        max_tokens=3000,
        stream=False
    )
    print(res.choices[0].message.content)

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
