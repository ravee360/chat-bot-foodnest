# 🚀 DocBot: Document Research & Theme Identifier

[![Docker](https://img.shields.io/badge/docker-enabled-blue)](https://docker.com)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://python.org)

A full-stack AI-powered system built for deep document research, intelligent theme identification, and citation-based answers. This system enables users to upload large sets of documents, ask natural language questions, and get context-aware, theme-driven answers with fine-grained citations, powered by Retrieval-Augmented Generation (RAG).

## 🌟 Key Features

### 📂 Document & Collection Management
- **Multi-format Support**: Upload and process PDFs, scanned images (PNG/JPG/TIFF)
- **Built-in OCR**: Tesseract integration for scanned/image-based documents
- **Smart Processing**: Rule-based chunking and text preprocessing for high-fidelity semantic search
- **Named Collections**: Create, select, and delete custom collections to group documents by domain, topic, or project
- **Isolated Storage**: Each collection is isolated within ChromaDB as its own semantic space

### 🤖 AI-Powered Research & Analysis
- **RAG Integration**: Retrieval-Augmented Generation with LLMs via OpenRouter API
- **Natural Language Queries**: Query over 75+ documents using conversational language
- **Theme Extraction**: Automatic identification of coherent themes from document responses
- **Granular Citations**: Includes document ID, page number, and paragraph for every extracted fact
- **Cross-document Synthesis**: Aggregates information from multiple documents for unified answers
- **Transparent Reasoning**: LLM Thought Process Panel shows detailed reasoning and understanding

### 🔍 Advanced Retrieval Techniques

#### ✅ Maximal Marginal Relevance (MMR)
- Balances relevance and diversity in retrieval
- Prevents redundant chunks from dominating results
- Ensures varied but contextually important information is surfaced

#### ✅ Cross-Encoder Re-Ranking
- Refines top-N retrieved chunks using transformer-based cross-encoder
- Deep token-level scoring for each query-passage pair
- Improves semantic relevance of final context passed to LLM

**Result**: Highly optimized context for LLMs that boosts factual accuracy, citation traceability, and response diversity.

### 💬 Frontend Interface
- **Modern UI**: Streamlit-based interface with intuitive sidebar navigation
- **Collection Management**: Easy creation, selection, and deletion of document collections
- **Drag-and-Drop Upload**: Automatic parsing and storage of uploaded files
- **Real-time Chat**: Interactive exploration through conversational interface
- **Collapsible Panels**: Organized display of themes, citations, and LLM analysis
- **Document Traceability**: Per-document result tables for comparison and verification

## 📊 Presentation Format

Fully aligned with professional research requirements:
- **Theme-wise Insights**: Grouped and categorized findings
- **Citation Breakdown**: Collapsible view for detailed source tracking
- **LLM Analysis**: Transparent thought process section
- **Document Mapping**: Tabular representation of individual document responses
- **Reference Tracking**: Visual mapping of document IDs supporting each insight

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Vector Store** | ChromaDB |
| **OCR Engine** | Tesseract |
| **LLM Access** | OpenRouter API |
| **Deployment** | Docker |

**Architecture**: Modular, scalable, and cleanly separated by service (parsing, RAG, vector storage, API)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Tesseract OCR (for image processing)
- OpenRouter API key (for LLM capabilities)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/doc_theme_bot.git
   cd doc_theme_bot
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Create a .env file in the project root
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```

### Running the Application

1. **Start the backend server**:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

2. **Start the frontend** (in a new terminal):
   ```bash
   streamlit run .\frontend\ui.py
   ```

3. **Access the application** at `http://localhost:8501`

## 💡 Usage Guide

### 1. Create a Collection
- Use the sidebar to create a new collection
- Collections help organize related documents by topic or project

### 2. Upload Documents
- Select your target collection
- Upload PDF or image files via drag-and-drop
- Monitor real-time processing status

### 3. Query Documents
- Type your question in the chat interface
- View AI-generated responses with themes
- Explore detailed evidence and citations
- Check document details and source mapping

## 📅 API Endpoints

### Document Management
- `POST /api/v1/documents/upload` - Upload a single document
- `POST /api/v1/documents/upload-multiple` - Upload multiple documents
- `GET /api/v1/documents/{doc_id}` - Get document details
- `DELETE /api/v1/documents/{doc_id}` - Delete a document

### Chat and Query
- `POST /api/v1/chat/query` - Query documents and get themes
  ```json
  {
    "query": "your question",
    "collection": "collection_name"
  }
  ```
  **Returns**: Answer, themes, evidence, and document details

### Collection Management
- `GET /api/v1/collections` - List all collections
- `POST /api/v1/collections` - Create a new collection
  ```json
  {
    "name": "collection_name"
  }
  ```
- `DELETE /api/v1/collections/{name}` - Delete a collection
- `GET /api/v1/collections/{name}` - Get collection details
- `GET /api/v1/collections/{name}/documents` - List documents in collection

### System
- `GET /` - Root endpoint with welcome message
- `GET /api/v1/openapi.json` - OpenAPI documentation

## 🌟 Advanced Capabilities

This system exceeds standard document processing by offering:

- **Multi-collection Management**: Organize documents across different projects
- **Fine-grained Citations**: Trace every fact to its source with precision
- **Explainable AI**: Transparent reasoning for all generated insights
- **Production Ready**: Suitable for legal, research, compliance, and policy domains

## 🔒 Security Features

- Secure API key management for LLM services
- Robust document processing with validation
- Input sanitization and validation
- Isolated collection storage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
