import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import requests
from rag_with_faiss import main

@pytest.fixture
def mock_get():
    with patch('rag_with_faiss.main.requests.get') as mock_get:
        yield mock_get

def test_fetch_sitemap(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url>
            <loc>http://example.com/page1</loc>
        </url>
        <url>
            <loc>http://example.com/page2</loc>
        </url>
    </urlset>
    """
    mock_get.return_value = mock_response

    urls = main.fetch_sitemap("http://example.com")
    assert urls == ["http://example.com/page1", "http://example.com/page2"]

def test_fetch_html(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><p>Hello, world!</p></body></html>"
    mock_get.return_value = mock_response

    html = main.fetch_html("http://example.com")
    assert html == "<html><body><p>Hello, world!</p></body></html>"

def test_chunk_text():
    text_list = ["This is a test sentence. " * 10]
    chunks = main.chunk_text(text_list, max_tokens=10)
    assert len(chunks) > 1

@patch('rag_with_faiss.main.client.embeddings.create')
def test_get_embedding(mock_create):
    # Create a mock object with an `embedding` attribute
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]  # ✅ Corrected

    # Create a mock response that contains the mock embedding object
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]  # ✅ Now it's a list of objects, not dicts

    mock_create.return_value = mock_response
    
    embedding = main.get_embedding("test text")
    assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    mock_create.assert_called_once_with(
        input="test text",
        model=main.EMBEDDING_DEPLOYMENT
    )

@patch('rag_with_faiss.main.faiss.read_index')
def test_load_faiss_index(mock_read_index):
    mock_index = MagicMock()
    mock_read_index.return_value = mock_index

    index = main.load_faiss_index("faiss_index.bin")
    assert index == mock_index

@patch('rag_with_faiss.main.faiss.write_index')
def test_save_faiss_index(mock_write_index):
    mock_index = MagicMock()
    main.save_faiss_index(mock_index, "faiss_index.bin")
    mock_write_index.assert_called_once_with(mock_index, "faiss_index.bin")

@patch('rag_with_faiss.main.pickle.load')
def test_load_text_chunks(mock_pickle_load):
    mock_pickle_load.return_value = [("chunk1", "url1"), ("chunk2", "url2")]
    with patch('builtins.open', new_callable=MagicMock) as mock_file:
        text_chunks = main.load_text_chunks("text_chunks.pkl")
        assert text_chunks == [("chunk1", "url1"), ("chunk2", "url2")]

@patch('rag_with_faiss.main.pickle.dump')
def test_save_text_chunks(mock_pickle_dump):
    text_chunks = [("chunk1", "url1"), ("chunk2", "url2")]
    
    # Create a mock file object that behaves like a context manager
    mock_file_handle = MagicMock()
    mock_open = MagicMock(return_value=mock_file_handle)
    
    mock_file_handle.__enter__.return_value = mock_file_handle
    with patch('builtins.open', mock_open):
        main.save_text_chunks(text_chunks, "text_chunks.pkl")
        
        # Verify open was called with correct arguments
        mock_open.assert_called_once_with("text_chunks.pkl", "wb")
        # Verify pickle.dump was called with correct arguments
        mock_pickle_dump.assert_called_once_with(text_chunks, mock_file_handle)

@patch('rag_with_faiss.main.fetch_html')
def test_scrape_website_less_then_30(mock_fetch_html):
    mock_fetch_html.return_value = """
    <html>
        <body>
            <h1>Main Heading</h1>
            <h2>Subheading</h2>
            <article>
                <p>First paragraph.</p>
                <p>Second paragraph.</p>
            </article>
        </body>
    </html>
    """
    text = main.scrape_website("http://example.com")
    assert text == []

@patch('rag_with_faiss.main.fetch_html')
def test_scrape_website_more_then_30(mock_fetch_html):
    mock_fetch_html.return_value = """
    <html>
        <body>
            <h1>Main Heading with more then 30 chars</h1>
            <h2>Subheading</h2>
            <article>
                <p>First paragraph with more then 30 chars</p>
                <p>Second paragraph.</p>
            </article>
        </body>
    </html>
    """
    text = main.scrape_website("http://example.com")
    assert text == ["Main Heading with more then 30 chars", "First paragraph with more then 30 chars"]


def test_chunk_website_content():
    text = "A" * 1200  # Simulating a long text
    url = "http://example.com"
    chunks = main.chunk_website_content(text, url, chunk_size=500)
    
    assert len(chunks) == 3  # Should create 3 chunks
    assert all(len(chunk[0]) <= 500 for chunk in chunks)  # Each chunk must be <= 500 chars
    assert all(chunk[1] == url for chunk in chunks)  # Ensure all chunks retain URL association

@patch('rag_with_faiss.main.get_embedding')
@patch('rag_with_faiss.main.faiss.IndexFlatL2.search')
def test_find_most_relevant(mock_search, mock_get_embedding):
    # Mock the get_embedding function
    mock_get_embedding.return_value = np.array([0.1, 0.2, 0.3])

    # Mock FAISS search() to return distances and indices
    mock_search.return_value = (
        np.array([[0.1, 0.2]]),  # Simulated distances (1 row, top_k columns)
        np.array([[0, 1]])       # Simulated indices matching text_chunks
    )

    text_chunks = [("Chunk1", "url1"), ("Chunk2", "url2")]
    
    # Mock FAISS index
    index = MagicMock()
    index.search = mock_search  # Ensure index.search() is properly mocked

    # Call the function
    results = main.find_most_relevant("test query", index, text_chunks)

    # Validate the results
    assert results == [("Chunk1", "url1"), ("Chunk2", "url2")]


@patch('rag_with_faiss.main.find_most_relevant')
def test_generate_answer_no_results(mock_find_most_relevant):
    mock_find_most_relevant.return_value = []  # Simulating no relevant results
    
    index = MagicMock()
    text_chunks = []
    response = main.generate_answer("What is AI?", index, text_chunks)
    
    assert response == "No relevant content found in the retrieved website text."

def test_load_faiss_index_invalid_file():
    with pytest.raises(FileNotFoundError):
        main.load_faiss_index("invalid_file.bin")