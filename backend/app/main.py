# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from backend.app.api import docs_api, chat_api
from backend.app.core.config import settings 

# Import services
from backend.app.services.vstore_svc import VectorStoreService
from backend.app.services.doc_parser import DocParserService as AccurateDocParserService # LLM based
from backend.app.services.doc_parser_fast import DocParserFastService
from backend.app.services.rag_svc import RAGService

from backend.app.api.collection_api import router as collection_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json", 
    version="0.1.0" 
)

origins = [
    "http://localhost",
    "http://localhost:8501", # Default Streamlit port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Include all routers with the correct prefix
app.include_router(docs_api.router, prefix=settings.API_V1_STR, tags=["Documents"])
app.include_router(chat_api.router, prefix=settings.API_V1_STR, tags=["Chat"])
app.include_router(collection_router, prefix=settings.API_V1_STR, tags=["Collections"])

# --- Singleton instances for services (application lifespan) ---
# These will be created by the Depends system on first use if not created here.
# Explicitly creating them on startup can help catch initialization errors early.
_vector_store_service_instance_main: VectorStoreService
_doc_parser_llm_instance_main: AccurateDocParserService
_doc_parser_fast_instance_main: DocParserFastService
_rag_service_instance_main: RAGService

@app.on_event("startup")
async def startup_event():
    """
    Initialize services when the application starts up.
    """
    print("FastAPI application starting up...")
    global _vector_store_service_instance_main, _doc_parser_llm_instance_main, \
           _doc_parser_fast_instance_main, _rag_service_instance_main
    
    print("Initializing services on startup...")
    _vector_store_service_instance_main = VectorStoreService()
    
    # Initialize both parsers
    _doc_parser_llm_instance_main = AccurateDocParserService(vector_store_service=_vector_store_service_instance_main)
    _doc_parser_fast_instance_main = DocParserFastService(vector_store_service=_vector_store_service_instance_main)
    
    _rag_service_instance_main = RAGService(vector_store_service=_vector_store_service_instance_main)
    
    print("VectorStoreService initialized:", "Yes" if _vector_store_service_instance_main._langchain_chroma_instance else "No")
    print("AccurateDocParserService (LLM) initialized:", "Yes" if _doc_parser_llm_instance_main else "No")
    print("DocParserFastService (Rule-based) initialized:", "Yes" if _doc_parser_fast_instance_main else "No")
    print("RAGService initialized:", "Yes" if _rag_service_instance_main else "No")
    
    print("FastAPI application startup complete.")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME} API!"}

# To run: uvicorn backend.app.main:app --reload --port 8000
