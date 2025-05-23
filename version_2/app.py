from flask import Flask, render_template, request, jsonify
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

class MovieSearchEngine:
    def __init__(self):
        self.db = None
        self.embedding_function = None
        self.initialize_db()
    
    def read_file_with_fallback(self, filepath):
        """Read file with multiple encoding attempts"""
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        print(f"Failed to decode {filepath}")
        return ""
    
    def initialize_db(self):
        """Initialize or load the ChromaDB database"""
        # Check if database already exists
        if os.path.exists("./chroma_subtitles"):
            print("Loading existing ChromaDB...")
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
            )
            self.db = Chroma(
                persist_directory="./chroma_subtitles",
                embedding_function=self.embedding_function
            )
        else:
            print("Creating new ChromaDB...")
            self.create_database()
    
    def create_database(self):
        """Create database from subtitle files"""
        subtitle_dir = r"C:\Users\abhay\OneDrive\Desktop\PJ_Revamp\Advanced-Semantic-Search-Engine\version_2\subtitles"
        
        if not os.path.exists(subtitle_dir):
            print(f"Warning: Subtitle directory {subtitle_dir} not found!")
            # Create empty database for demo purposes
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
            )
            self.db = Chroma(
                persist_directory="./chroma_subtitles",
                embedding_function=self.embedding_function
            )
            return
        
        documents = []
        
        # Read all subtitle files
        for filename in os.listdir(subtitle_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(subtitle_dir, filename)
                text = self.read_file_with_fallback(filepath)
                if text:
                    documents.append(Document(
                        page_content=text, 
                        metadata={"source": filename}
                    ))
        
        if not documents:
            print("No subtitle files found!")
            return
        
        # Split documents into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for doc in documents:
            splits = splitter.split_text(doc.page_content)
            for chunk in splits:
                docs.append(Document(
                    page_content=chunk, 
                    metadata={"source": doc.metadata["source"]}
                ))
        
        # Create embeddings and database
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        self.db = Chroma.from_documents(
            docs, 
            embedding=self.embedding_function, 
            persist_directory="./chroma_subtitles"
        )
        self.db.persist()
        print(f"Database created with {len(docs)} chunks from {len(documents)} movies")
    
    def search_movies(self, subtitle_line, top_k=5):
        """Search for movies based on subtitle line"""
        if not self.db:
            return {"error": "Database not initialized"}
        
        try:
            results = self.db.similarity_search_with_score(subtitle_line, k=top_k)
            
            search_results = []
            for doc, score in results:
                search_results.append({
                    "movie": doc.metadata['source'].replace('.txt', ''),
                    "score": round(score, 4),
                    "matched_text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "confidence": round((1 - score) * 100, 1)  # Convert distance to confidence percentage
                })
            
            return {"results": search_results}
        
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

# Initialize the search engine
search_engine = MovieSearchEngine()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """API endpoint for searching movies"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    top_k = data.get('top_k', 5)
    
    results = search_engine.search_movies(query, top_k)
    return jsonify(results)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database_initialized": search_engine.db is not None
    })

if __name__ == '__main__':
    print("Starting Movie Subtitle Search Engine...")
    print("Open your browser and go to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)