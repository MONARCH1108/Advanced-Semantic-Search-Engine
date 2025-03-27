# 🎥 **SubtitleSense: Advanced Semantic Search Engine**

**SubtitleSense** is an advanced subtitle search engine that leverages **OpenAI Whisper**, **ChromaDB**, and **NLP models** for accurate semantic search, filtering, and visualizations. It also includes audio transcription capabilities, making it a powerful tool for subtitle analysis and retrieval.

---

## 🚀 **Features**

- ✅ **Semantic Search:**  
  - Perform accurate subtitle search using sentence embeddings for improved relevance.  
  - Returns matching subtitles with similarity scores and rich metadata.

- ✅ **Audio Transcription:**  
  - Convert audio files into text using **Whisper**, enabling subtitle extraction from audio content.

- ✅ **Advanced Filtering:**  
  - Filter subtitles by genre, year, language, and sentiment score.  
  - Combine filtering with semantic search for precise results.  

- ✅ **Data Visualization:**  
  - Generate visual insights, such as **sentiment distribution by genre** and **movie count by year**.  

- ✅ **ChromaDB for Embeddings:**  
  - Efficiently stores and retrieves subtitle embeddings for fast and scalable search.

---

## 🛠️ **Tech Stack**

- **Python**: Main programming language.  
- **SQLite**: In-memory database for storing subtitle data.  
- **ChromaDB**: Vector database for fast embedding retrieval.  
- **Whisper**: OpenAI model for audio transcription.  
- **Sentence-Transformers**: For generating semantic embeddings.  
- **Matplotlib & Seaborn**: For data visualizations.  

---

## 📦 **Installation**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/MONARCH1108/Advanced-Semantic-Search-Engine
    cd SubtitleSense
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Install Whisper:**
    ```bash
    pip install openai-whisper
    ```

> 💡 Ensure you have **ffmpeg** installed for audio processing:
    ```bash
    sudo apt install ffmpeg  # Linux  
    brew install ffmpeg      # macOS  
    ```

---

## ⚙️ **Usage**

### 1. **Run the Application**
```bash
python AI_SEO.py

