import flask
from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time

app=Flask(__name__)
model = None
collection = None

def scrape_capitals():
    URL = "https://en.wikipedia.org/wiki/List_of_national_capitals"
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all("table", {"class": "wikitable"})
    
    capital_data=[]
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) >= 3:
                country = cols[0].get_text(strip=True)
                capital = cols[1].get_text(strip=True)
                continent = cols[2].get_text(strip=True)
                capital_data.append(f"{country}-{capital}-{continent}")
    return capital_data

def build_chromadb(documents, model):
    client = Client(Settings())
    collection = client.create_collection(name="capital_qa")
    ids = [f"cap_{i}" for i in range(len(documents))]
    print("Encoding documents (embedding)...")
    start = time.time()
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True).tolist()
    #print(f"Embedding done in {time.time() - start:.2f}s")
    print("\nRetrieved docs:")
    collection.add(documents=documents, ids=ids, embeddings=embeddings)
    return collection

def generate_answer(query, docs):
    query_lower = query.lower()
    for doc in docs:
        try:
            capital, country, continent = doc.strip().split("-", maxsplit=2)
        except ValueError:
            continue

        # Normalize
        country = country.strip().lower()
        capital = capital.strip().lower()
        continent = continent.strip().lower()

        if "capital" in query_lower and country in query_lower:
            return f"The capital of {country.title()} is {capital.title()}."
        elif "which country" in query_lower and capital in query_lower:
            return f"{capital.title()} is the capital of {country.title()}."
        elif "continent" in query_lower and (country in query_lower or capital in query_lower):
            return f"{country.title()} is in {continent.title()}."
    return "Sorry, I couldn't find a confident answer."

###############################################################################################
def initialize_app():
    global model, collection
    print("📡 Scraping and initializing...")
    capital_data = scrape_capitals()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = build_chromadb(capital_data, model)

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, collection
    answer = None
    retrieved = []

    if request.method == 'POST':
        query = request.form.get('query')
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding,n_results=1)
        docs = results['documents'][0]
        answer = generate_answer(query, docs)
        retrieved = docs

    return render_template('index.html', answer=answer, retrieved=retrieved)

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True)

'''
def main():
    print("Scraping data...")
    capital_data = scrape_capitals()
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    print("Building vector store...")
    collection = build_chromadb(capital_data, model)
    query = input("Ask a question about a capital (e.g., What is the capital of Kenya?): ")
    print("Retrieving relevant documents...")
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    retrieved_docs = results['documents'][0]
    answer = generate_answer(query, retrieved_docs)
    print("\n Answer:", answer)

if __name__ == "__main__":
    main()
'''