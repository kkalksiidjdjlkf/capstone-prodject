"""
FastAPI Backend for KREPS RAG System
Handles API requests, authentication, and business logic
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from datetime import datetime
import os
import logging
import uuid
import json

# Import RAG components
from src.database.client import DatabaseClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.reranker import Reranker
from src.response.response_generator import ResponseGenerator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from initialize_rag import RAGInitializer, RAGComponents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KREPS RAG API",
    description="Backend API for KREPS RAG System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_client():
    """Initialize database client with connection pooling"""
    db_config = {
        'url': os.getenv('DATABASE_URL', 'postgresql://rag_user:rag_password@localhost:5432/rag_system'),
        'pool_size': int(os.getenv('DATABASE_POOL_SIZE', '5')),
        'max_connections': int(os.getenv('DATABASE_MAX_CONNECTIONS', '20')),
        'timeout': int(os.getenv('DATABASE_TIMEOUT', '10'))
    }
    return DatabaseClient(db_config)

# RAG components
def get_rag_components():
    """Initialize RAG components"""
    db_client = get_db_client()
    
    # Load configuration
    import yaml
    with open('config/process_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    retriever = HybridRetriever(config=config)
    reranker = Reranker(config=config)
    generator = ResponseGenerator(config=config)
    return db_client, retriever, reranker, generator

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting KREPS RAG Backend...")
    try:
        db_client = get_db_client()
        if db_client.health_check():
            logger.info("Database connection established")
        else:
            logger.warning("Database connection failed")
    except Exception as e:
        logger.error(f"Startup error: {e}")

# API Endpoints
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    sensitivity_level: str = "internal",
    language: Optional[str] = None
):
    """
    Upload and process a document

    Args:
        file: Document file to upload
        sensitivity_level: Security level (public, internal, confidential)
        language: Document language (auto, ko, en, mixed)

    Returns:
        Document processing status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Save file
        file_path = f"data/raw/{file.filename}"
        os.makedirs("data/raw", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Saved file: {file_path}")

        # Process document (async)
        # In production, this would be a background task
        doc_id = str(uuid.uuid4())
        file_type = file.filename.split('.')[-1].lower()

        # Create document record
        db_client = get_db_client()
        doc_id = db_client.create_document(
            file_name=file.filename,
            file_path=file_path,
            file_type=file_type,
            language=language or "auto",
            sensitivity_level=sensitivity_level,
            file_hash=str(uuid.uuid4())  # Simplified for demo
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "file_name": file.filename,
            "message": "Document uploaded successfully. Processing will complete shortly."
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.post("/api/query")
async def process_query(
    query_data: Dict
):
    """
    Process a RAG query

    Args:
        query_data: {
            "query": "user question",
            "user_id": "optional user identifier",
            "language": "optional language preference"
        }

    Returns:
        Query results with sources
    """
    try:
        user_query = query_data.get("query")
        user_id = query_data.get("user_id", "anonymous")
        language = query_data.get("language", "auto")

        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required")

        logger.info(f"Processing query: {user_query[:50]}...")

        # Initialize RAG components
        db_client, retriever, reranker, generator = get_rag_components()

        # Process query
        start_time = datetime.now()

        # Retrieve documents
        retrieved_docs = retriever.retrieve(user_query)

        # Rerank documents
        reranked_docs = reranker.rerank(user_query, retrieved_docs)

        # Generate response
        response = generator.generate_answer(user_query, reranked_docs)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log query
        db_client.log_query(
            original_query=user_query,
            detected_language=language,
            llm_response=response,
            processing_time_ms=int(processing_time),
            retrieved_chunk_count=len(retrieved_docs),
            avg_similarity_score=reranked_docs[0]["score"] if reranked_docs else 0.0,
            user_id=user_id
        )

        return {
            "status": "success",
            "query": user_query,
            "response": response,
            "sources": [doc["metadata"] for doc in reranked_docs[:5]],
            "processing_time_ms": processing_time,
            "retrieved_documents": len(retrieved_docs),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    """
    Get system analytics

    Returns:
        System usage statistics
    """
    try:
        db_client = get_db_client()
        stats = db_client.get_system_stats()

        # Get recent queries
        recent_queries = db_client.get_recent_queries(10)

        return {
            "status": "success",
            "analytics": {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "queries_24h": stats.get("queries_24h", 0),
                "documents_by_language": stats.get("documents_by_language", {}),
                "recent_queries": [
                    {
                        "query": q["original_query"][:50] + "...",
                        "language": q["detected_language"],
                        "time": q["created_at"].isoformat()
                    }
                    for q in recent_queries
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")

# Global RAG system state
rag_system_state = {
    "initialized": False,
    "components": None,
    "last_init_time": None,
    "error": None
}

@app.post("/api/rag/initialize")
async def initialize_rag_system(
    force: bool = False,
    incremental: bool = True
):
    """
    Initialize or reinitialize the RAG system

    Args:
        force: Force complete rebuild (ignores cache)
        incremental: Use incremental rebuild (only process changed files)

    Returns:
        Initialization status and system information
    """
    global rag_system_state

    try:
        # Check if already initialized and not forcing rebuild
        if rag_system_state["initialized"] and not force:
            return {
                "status": "success",
                "message": "RAG system already initialized",
                "initialized": True,
                "last_init_time": rag_system_state["last_init_time"],
                "chunks_count": len(rag_system_state["components"].original_chunks) if rag_system_state["components"] else 0
            }

        logger.info(f"{'Force rebuilding' if force else 'Initializing'} RAG system...")

        # Clear previous error state
        rag_system_state["error"] = None

        # Initialize RAG system
        initializer = RAGInitializer("config/init_config.yaml", "config/process_config.yaml")

        if force:
            # Force full rebuild - create fresh components
            components = initializer.initialize()
            components.retriever.initialize(components.original_chunks)
        elif incremental and rag_system_state["components"] is None:
            # First time initialization with incremental logic
            components = initializer.initialize()
            components.retriever.initialize(components.original_chunks)
        else:
            # Try incremental rebuild
            try:
                components = initializer.initialize()  # Uses incremental logic internally
                components.retriever.initialize(components.original_chunks)
            except Exception as e:
                logger.warning(f"Incremental rebuild failed: {str(e)}, falling back to full rebuild")
                # Fall back to full rebuild
                components = initializer.initialize()
                components.retriever.initialize(components.original_chunks)

        # Update global state
        rag_system_state["initialized"] = True
        rag_system_state["components"] = components
        rag_system_state["last_init_time"] = datetime.now().isoformat()

        chunks_count = len(components.original_chunks)
        logger.info(f"RAG system initialized successfully with {chunks_count} chunks")

        return {
            "status": "success",
            "message": f"RAG system {'rebuilt' if force else 'initialized'} successfully",
            "initialized": True,
            "chunks_count": chunks_count,
            "last_init_time": rag_system_state["last_init_time"],
            "force_rebuild": force,
            "incremental": incremental
        }

    except Exception as e:
        error_msg = f"RAG initialization failed: {str(e)}"
        logger.error(error_msg)

        # Update error state
        rag_system_state["error"] = str(e)
        rag_system_state["initialized"] = False

        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/rag/status")
async def get_rag_status():
    """
    Get current RAG system status

    Returns:
        Current initialization status and system information
    """
    global rag_system_state

    if rag_system_state["initialized"] and rag_system_state["components"]:
        components = rag_system_state["components"]
        return {
            "status": "success",
            "initialized": True,
            "chunks_count": len(components.original_chunks),
            "last_init_time": rag_system_state["last_init_time"],
            "system_info": {
                "retriever_type": type(components.retriever).__name__,
                "reranker_type": type(components.reranker).__name__,
                "generator_type": type(components.response_generator).__name__,
                "embedding_model": components.process_config.get("model", {}).get("embedding_model_hf", "unknown")
            },
            "error": None
        }
    else:
        return {
            "status": "success",
            "initialized": False,
            "chunks_count": 0,
            "last_init_time": rag_system_state["last_init_time"],
            "system_info": None,
            "error": rag_system_state["error"]
        }

@app.post("/api/rag/reinitialize")
async def reinitialize_rag_system():
    """
    Force reinitialize the entire RAG system

    Returns:
        Reinitialization status
    """
    global rag_system_state

    logger.info("Force reinitializing RAG system...")

    # Reset state
    rag_system_state["initialized"] = False
    rag_system_state["components"] = None
    rag_system_state["error"] = None

    # Reinitialize
    return await initialize_rag_system(force=True, incremental=False)

@app.get("/api/rag/documents")
async def get_indexed_documents():
    """
    Get information about indexed documents

    Returns:
        List of indexed documents with metadata
    """
    global rag_system_state

    if not rag_system_state["initialized"] or not rag_system_state["components"]:
        # Return empty list instead of error when not initialized
        return {
            "status": "success",
            "total_documents": 0,
            "total_chunks": 0,
            "documents": []
        }

    components = rag_system_state["components"]

    # Group chunks by source and collect statistics
    doc_stats = {}
    for chunk in components.original_chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in doc_stats:
            doc_stats[source] = {
                "source": source,
                "chunk_count": 0,
                "total_chars": 0,
                "file_type": source.split('.')[-1] if '.' in source else "unknown"
            }
        doc_stats[source]["chunk_count"] += 1
        doc_stats[source]["total_chars"] += len(chunk.page_content)

    return {
        "status": "success",
        "total_documents": len(doc_stats),
        "total_chunks": len(components.original_chunks),
        "documents": list(doc_stats.values())
    }

@app.post("/api/query")
async def process_query_enhanced(
    query_data: Dict
):
    """
    Enhanced RAG query processing using initialized system

    Args:
        query_data: {
            "query": "user question",
            "use_fusion": true/false,
            "top_k": 5
        }

    Returns:
        Query results with sources
    """
    global rag_system_state

    if not rag_system_state["initialized"] or not rag_system_state["components"]:
        # Return error message instead of 400 when not initialized
        return {
            "status": "error",
            "message": "RAG system not initialized. Please initialize first.",
            "query": user_query,
            "response": "The RAG system needs to be initialized before processing queries. Please go to the Ingestion tab and initialize the system.",
            "sources": [],
            "processing_time_ms": 0,
            "retrieved_documents": 0,
            "fusion_used": use_fusion,
            "timestamp": datetime.now().isoformat()
        }

    try:
        user_query = query_data.get("query")
        use_fusion = query_data.get("use_fusion", False)
        top_k = query_data.get("top_k", 5)

        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required")

        logger.info(f"Processing enhanced query: {user_query[:50]}...")

        components = rag_system_state["components"]
        start_time = datetime.now()

        # Use fusion if requested and enabled
        if use_fusion and components.process_config.get('fusion', {}).get('enabled', False):
            retrieved_docs = components.retriever.retrieve_with_fusion(user_query, top_k=top_k)
        else:
            retrieved_docs = components.retriever.retrieve_with_method(user_query, "hybrid", top_k=top_k)

        # Convert to expected format for reranking and response generation
        doc_objects = [doc.document for doc in retrieved_docs]

        # Rerank documents
        reranked_docs = components.reranker.rerank(user_query, doc_objects)

        # Generate response
        response = components.response_generator.generate_answer(user_query, reranked_docs)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "status": "success",
            "query": user_query,
            "response": response,
            "sources": [doc.metadata for doc in reranked_docs[:5]],
            "processing_time_ms": processing_time,
            "retrieved_documents": len(retrieved_docs),
            "fusion_used": use_fusion,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Enhanced query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_client = get_db_client()
        db_status = "healthy" if db_client.health_check() else "unhealthy"

        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
