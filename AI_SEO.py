import sqlite3
import pandas as pd
import numpy as np
import re
import os
import speech_recognition as sr
import soundfile as sf
import whisper
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedSubtitleSearchEngine:
    def __init__(self, sample_percentage: float = 1.0):
        """
        Initialize the Advanced Subtitle Search Engine
        
        :param sample_percentage: Percentage of data to use
        """
        self.sample_percentage = sample_percentage
        
        # Create SQLite database in memory
        self.conn = sqlite3.connect(':memory:')
        
        # Initialize models
        self._initialize_models()
        
        # Create sample database
        self._create_sample_database()
    
    def _initialize_models(self):
        """
        Initialize various NLP and ML models
        """
        # Semantic embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Speech recognition model (Whisper)
        self.whisper_model = whisper.load_model("base")
        
        # Speech recognition 
        self.recognizer = sr.Recognizer()
        
        # ChromaDB for embedding storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.create_collection(name="subtitle_embeddings")
    
    def _create_sample_database(self):
        """
        Create a comprehensive sample subtitle database
        """
        cursor = self.conn.cursor()
        
        # Create enhanced subtitles table
        cursor.execute('''
            CREATE TABLE subtitles (
                id INTEGER PRIMARY KEY,
                movie_title TEXT,
                genre TEXT,
                year INTEGER,
                subtitle_text TEXT,
                language TEXT,
                sentiment REAL
            )
        ''')
        
        # Sample data with more comprehensive information
        sample_data = [
            {
                'movie_title': 'The Shawshank Redemption',
                'genre': 'Drama',
                'year': 1994,
                'subtitles': [
                    "Hope is a good thing, maybe the best of things, and no good thing ever dies.",
                    "Get busy living, or get busy dying.",
                    "I have no idea to this day what those two Italian ladies were singing about."
                ],
                'language': 'English',
                'sentiments': [0.8, 0.7, 0.5]
            },
            {
                'movie_title': 'Inception',
                'genre': 'Sci-Fi',
                'year': 2010,
                'subtitles': [
                    "An idea is like a virus, resilient, highly contagious.",
                    "Dreams feel real while we're in them.",
                    "Do you want to take a leap of faith?"
                ],
                'language': 'English',
                'sentiments': [0.6, 0.7, 0.8]
            }
        ]
        
        # Insert sample data
        for movie in sample_data:
            for subtitle, sentiment in zip(movie['subtitles'], movie['sentiments']):
                cursor.execute('''
                    INSERT INTO subtitles 
                    (movie_title, genre, year, subtitle_text, language, sentiment) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    movie['movie_title'], 
                    movie['genre'], 
                    movie['year'], 
                    subtitle, 
                    movie['language'], 
                    sentiment
                ))
        
        self.conn.commit()
        print("Enhanced subtitle database created successfully!")
    
    def audio_to_text(self, audio_path: str) -> str:
        """
        Convert audio to text using Whisper
        
        :param audio_path: Path to audio file
        :return: Transcribed text
        """
        # Transcribe audio
        result = self.whisper_model.transcribe(audio_path)
        return result['text']
    
    def ingest_subtitles(self):
        """
        Ingest subtitles from the database, clean and embed
        """
        # Read subtitles
        df = pd.read_sql_query("SELECT * FROM subtitles", self.conn)
        
        # Sample data if needed
        if self.sample_percentage < 1:
            df = df.sample(frac=self.sample_percentage)
        
        # Process and embed subtitles
        for index, row in df.iterrows():
            # Clean and embed subtitle
            cleaned_text = self._clean_text(row['subtitle_text'])
            embedding = self.embedding_model.encode(cleaned_text)
            
            # Store in ChromaDB with rich metadata
            unique_id = f"{row['id']}_{index}"
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[cleaned_text],
                ids=[unique_id],
                metadatas=[{
                    'movie_title': row['movie_title'],
                    'genre': row['genre'],
                    'year': row['year'],
                    'language': row['language'],
                    'sentiment': row['sentiment']
                }]
            )
        
        print("Subtitle ingestion complete!")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        :param text: Input text
        :return: Cleaned text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic search on subtitles
        
        :param query: Search query
        :param top_k: Number of top results
        :return: List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for doc, distance, metadata in zip(
            results['documents'][0], 
            results['distances'][0], 
            results['metadatas'][0]
        ):
            formatted_results.append({
                'subtitle': doc,
                'similarity_score': 1 - distance,
                'movie_title': metadata['movie_title'],
                'genre': metadata['genre'],
                'year': metadata['year'],
                'language': metadata['language'],
                'sentiment': metadata.get('sentiment', None)
            })
        
        return formatted_results
    
    def advanced_filtering(self, 
                            query: str = None, 
                            genre: str = None, 
                            min_year: int = None, 
                            max_year: int = None, 
                            language: str = None,
                            min_sentiment: float = None,
                            max_sentiment: float = None,
                            top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Advanced filtering of subtitles with optional semantic search
        
        :param query: Optional semantic search query
        :param genre: Filter by genre
        :param min_year: Minimum year
        :param max_year: Maximum year
        :param language: Filter by language
        :param min_sentiment: Minimum sentiment score
        :param max_sentiment: Maximum sentiment score
        :param top_k: Number of top results
        :return: Filtered and optionally semantically searched results
        """
        # Start with base query
        base_query = "SELECT * FROM subtitles WHERE 1=1"
        params = []
        
        # Add filters
        if genre:
            base_query += " AND genre = ?"
            params.append(genre)
        
        if min_year:
            base_query += " AND year >= ?"
            params.append(min_year)
        
        if max_year:
            base_query += " AND year <= ?"
            params.append(max_year)
        
        if language:
            base_query += " AND language = ?"
            params.append(language)
        
        if min_sentiment is not None:
            base_query += " AND sentiment >= ?"
            params.append(min_sentiment)
        
        if max_sentiment is not None:
            base_query += " AND sentiment <= ?"
            params.append(max_sentiment)
        
        # Execute filtered query
        df = pd.read_sql_query(base_query, self.conn, params=params)
        
        # If semantic query provided, re-rank results
        if query:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Compute semantic similarities
            similarities = []
            for _, row in df.iterrows():
                subtitle_embedding = self.embedding_model.encode(row['subtitle_text'])
                similarity = np.dot(query_embedding, subtitle_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(subtitle_embedding)
                )
                similarities.append(similarity)
            
            # Add similarities to dataframe and sort
            df['semantic_score'] = similarities
            df = df.sort_values('semantic_score', ascending=False).head(top_k)
        
        # Convert to list of dictionaries
        results = df.to_dict('records')
        return results
    
    def generate_visualizations(self):
        """
        Generate visualizations of subtitle data
        """
        # Read full dataset
        df = pd.read_sql_query("SELECT * FROM subtitles", self.conn)
        
        # Visualization 1: Distribution of Sentiments by Genre
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='genre', y='sentiment', data=df)
        plt.title('Sentiment Distribution by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Sentiment Score')
        plt.tight_layout()
        plt.savefig('sentiment_by_genre.png')
        plt.close()
        
        # Visualization 2: Movies by Year
        plt.figure(figsize=(10, 6))
        df['year'].value_counts().sort_index().plot(kind='bar')
        plt.title('Number of Movies by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.tight_layout()
        plt.savefig('movies_by_year.png')
        plt.close()
        
        print("Visualizations generated successfully!")

def main():
    # Initialize advanced search engine
    search_engine = AdvancedSubtitleSearchEngine()
    
    # Ingest subtitles
    search_engine.ingest_subtitles()
    
    # Demonstrate semantic search
    print("\n--- Semantic Search ---")
    semantic_results = search_engine.semantic_search("hope and inspiration")
    for result in semantic_results:
        print(f"Movie: {result['movie_title']}")
        print(f"Subtitle: {result['subtitle']}")
        print(f"Similarity Score: {result['similarity_score']:.2f}\n")
    
    # Demonstrate advanced filtering
    print("\n--- Advanced Filtering ---")
    filtered_results = search_engine.advanced_filtering(
        query="dreams and reality",
        genre="Sci-Fi",
        min_year=2000,
        max_year=2020,
        min_sentiment=0.6
    )
    for result in filtered_results:
        print(f"Movie: {result['movie_title']}")
        print(f"Subtitle: {result['subtitle_text']}")
        print(f"Sentiment: {result['sentiment']}\n")
    
    # Generate visualizations
    search_engine.generate_visualizations()
    
    # Optional: Audio to text conversion (commented out as it requires an audio file)
    # text = search_engine.audio_to_text('path/to/audio/file.wav')
    # print("Transcribed Text:", text)

if __name__ == "__main__":
    main()