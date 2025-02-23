import os
import pickle
import urllib
import xml.etree.ElementTree as ET

import faiss
import numpy as np
import openai
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss_index.bin")
TEXT_CHUNKS_FILE = os.getenv("TEXT_CHUNKS_FILE", "text_chunks.pkl")
SITE_URL = os.getenv("SITE_URL")

# Check if the index should be regenerated or loaded
LOAD_EXISTING_INDEX = os.getenv("LOAD_EXISTING_INDEX", "false").lower() == "true"  # New variable to control loading


# Initialize OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

visited_urls = set()

def fetch_sitemap(url):
    """Fetch and parse the sitemap.xml from the website, ignoring the namespace."""
    sitemap_url = url.rstrip("/") + "/sitemap.xml"  # Assuming sitemap is at /sitemap.xml
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        sitemap_xml = response.text
        
        # Parse the sitemap XML
        root = ET.fromstring(sitemap_xml)

        # Strip namespaces from tags using a helper function
        def strip_namespace(tag):
            """Remove the namespace from an XML tag."""
            return tag.split('}')[1] if '}' in tag else tag

        urls = []

        # Find all <loc> tags using a loop that ignores the namespace
        for elem in root.iter():
            if strip_namespace(elem.tag) == 'loc':  # Compare tag names without namespaces
                urls.append(elem.text)

        return urls
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []


def crawl_website_using_sitemap(start_url):
    """Crawl the website by parsing its sitemap.xml."""
    print(f"Fetching sitemap from {start_url}...")
    urls = fetch_sitemap(start_url)

    if not urls:
        print("No URLs found in sitemap or failed to fetch sitemap.")
        return []

    print(f"Found {len(urls)} URLs in sitemap.")
    return urls

def fetch_html(url):
    """Fetches HTML content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_website(url):
    """Extracts and returns main content from a webpage."""
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Extract meaningful text
    text_elements = []

    # Grab <h1>, <h2> (important headings)
    for header in soup.find_all(["h1", "h2"]):
        text_elements.append(header.get_text(strip=True))

    # Grab paragraphs inside <article> or <main>
    main_content = soup.find(["main", "article"])
    if main_content:
        text_elements.extend([p.get_text(strip=True) for p in main_content.find_all("p")])

    # Fallback: Get all <p> tags (if main/article missing)
    if not text_elements:
        text_elements.extend([p.get_text(strip=True) for p in soup.find_all("p")])

    # Remove junk (short texts like "Home", "Login", etc.)
    return [text for text in text_elements if len(text) > 30]

def scrape_pages(urls):
    """Scrape content from each page and return text."""
    all_texts = []
    
    for url in urls:
        print(f"Scraping {url}...")
        texts = scrape_website(url)  # Use the existing scrape function
        if texts:
            all_texts.extend(texts)
    
    return all_texts

def index_website_content(all_texts):
    """Generate embeddings and index content from all pages."""
    print("Generating embeddings...")
    embeddings = np.array([get_embedding(chunk) for chunk in tqdm(all_texts)])

    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Save FAISS index to disk
    save_faiss_index(index)

    return index

def get_embedding(text):
    """Generates and returns an embedding for the given text."""
    response = client.embeddings.create(input=text, model=EMBEDDING_DEPLOYMENT)
    # return np.array(response.data[0].embedding)
    return np.array(response.data[0]['embedding'])

def chunk_text(text_list, max_tokens=300):
    """Splits text into smaller chunks to ensure efficient processing."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    return [
        tokenizer.decode(tokens[i:i+max_tokens])
        for text in text_list
        for tokens in [tokenizer.encode(text)]
        for i in range(0, len(tokens), max_tokens)
    ]

def create_faiss_index(embeddings):
    """Creates and returns a FAISS index from embeddings."""
    if embeddings.size == 0:
        raise ValueError("No embeddings found to index.")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Load FAISS index from disk
def load_faiss_index(filepath=FAISS_INDEX_FILE):
    """Loads a FAISS index from disk."""
    try:
        index = faiss.read_index(filepath)
        print(f"FAISS index loaded successfully from {filepath}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # raise FileNotFoundError error instead of returning None
        raise FileNotFoundError(f"Error loading FAISS index: {e}")
        return None

# Save FAISS index to disk
def save_faiss_index(index, filepath=FAISS_INDEX_FILE):
    """Saves the FAISS index to disk."""
    faiss.write_index(index, filepath)
    print(f"FAISS index saved to {filepath}")

def chunk_website_content(text, url, chunk_size=500):
    """Chunk website content into smaller pieces of text, with their URLs."""
    # Split the text into chunks (e.g., sentences or paragraphs)
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Return the chunks with their corresponding URL
    return [(chunk, url) for chunk in text_chunks]

# Load the text chunks from disk
def load_text_chunks(filepath=TEXT_CHUNKS_FILE):
    """Loads the text chunks from disk."""
    try:
        with open(filepath, 'rb') as f:
            text_chunks = pickle.load(f)
        print(f"Text chunks loaded successfully from {filepath}")
        return text_chunks
    except Exception as e:
        print(f"Error loading text chunks: {e}")
        return []

# Save the text chunks to disk
def save_text_chunks(text_chunks, filepath=TEXT_CHUNKS_FILE):
    """Saves the text chunks to disk."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(text_chunks, f)
        print(f"Text chunks saved to {filepath}")
    except Exception as e:
        print(f"Error saving text chunks: {e}")

def find_most_relevant(query, index, text_chunks, top_k=5):
    """Finds the most relevant text chunks for a query using FAISS."""
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    # Ensure indices are within bounds
    relevant_chunks = []
    for i in indices[0]:
        if i < len(text_chunks):  # Only add valid indices
            chunk, url = text_chunks[i]  # Get chunk and its URL
            relevant_chunks.append((chunk, url))
    
    return relevant_chunks if relevant_chunks else [("No relevant content found.", "")]

def generate_answer(query, index, text_chunks):
    """Uses Azure OpenAI to generate an answer based on retrieved context."""
    relevant_chunks = find_most_relevant(query, index, text_chunks)


    # Flatten the list of chunks if any chunk is a list of strings.
    flattened_chunks = []
    for chunk, url in relevant_chunks:
        # If the chunk is a list of strings, flatten it
        if isinstance(chunk, list):
            flattened_chunks.extend(chunk)
        elif isinstance(chunk, str):
            flattened_chunks.append(chunk)

    # Ensure that relevant_chunks contains tuples of (chunk, url)
    context = "\n\n".join([str(chunk).strip() for chunk in flattened_chunks if chunk and isinstance(chunk, str)])
 
    # If no valid content, return a message indicating no relevant content was found
    if not context:
        return "No relevant content found in the retrieved website text."

    # Refined prompt to guide the model
    prompt = f"""
    You are an AI assistant that provides informative answers based on website content.

    Here is relevant content extracted from the website:
    -------------------
    {context}
    -------------------

    Based on the information above, answer the following question:
    {query}
    Be concise and focus on the main topic or key points.
    """

    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Answer questions based only on the given website content."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    # Add URLs of the relevant chunks to the answer
    url_references = "\n".join([f"Source: {url}" for _, url in relevant_chunks if url])
    return f"{answer}\n\n{url_references}"

def main():
    """Main function to scrape, embed, index, and answer queries."""

    # Step 1: Load the FAISS index if the environment variable is set
    if LOAD_EXISTING_INDEX:
        print("Loading existing FAISS index and text chunks...")
        index = load_faiss_index()
        all_texts_with_urls = load_text_chunks()
    else:
        # Step 1: Crawl the website to get URLs
        print("Crawling the website...")
        all_urls = crawl_website_using_sitemap(SITE_URL)

        # Step 2: Scrape content from each URL
        print("Scraping website content...")
        all_texts_with_urls = []

        for url in all_urls:
            # Discard url that contains "category" or "tag"
            if "category" in url or "tag" in url:
                continue
            print(f"Scraping {url}...")
            texts = scrape_website(url)  # Use the existing scrape function
            if texts:
                all_texts_with_urls.extend(chunk_website_content(texts, url))  # Include URLs in chunks
        
        # Step 3: Create FAISS index from scraped content
        print("Indexing content...")

        # Without tqdm
        # embeddings = np.array([get_embedding(chunk) for chunk, _ in all_texts_with_urls])

        # Now with tqdm
        embeddings = np.array([get_embedding(chunk) for chunk, _ in tqdm(all_texts_with_urls, desc="Generating embeddings")])
        index = create_faiss_index(embeddings)

        # Save FAISS index and text chunks to disk
        save_faiss_index(index)
        save_text_chunks(all_texts_with_urls)

    # Step 4: Answer a question based on the indexed content
    question = "What is the main topic of the website?"
    answer = generate_answer(question, index, all_texts_with_urls)
    print("Answer:", answer)

if __name__ == "__main__":
    main()

    # -----------------------------------------
    # How to Load FAISS Index from Disk Later:
    # -----------------------------------------
    # To reuse the FAISS index in a new script:
    #
    # index = load_faiss_index()
    # if index:
    #     answer = generate_answer("Your question here", index, text_chunks)
    #     print(answer)
