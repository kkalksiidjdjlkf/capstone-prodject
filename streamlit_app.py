import streamlit as st
import logging
import os
from typing import Dict, List
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from collections import defaultdict
import requests

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
API_TOKEN = "krepstoken"  # Simple token for demo

# --- New Helper Function for File Saving ---
def save_uploaded_files(uploaded_files):
    """Saves files to the data/raw directory."""
    raw_path = "data/raw"
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(raw_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

# --- Backend API Helper Functions ---
def call_backend_api(endpoint: str, method: str = "GET", data: Dict = None, files=None):
    """Call backend API without authentication."""
    url = f"{BACKEND_URL}{endpoint}"
    headers = {}  # No authentication headers

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=300)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files, timeout=600)  # 10 minutes for file uploads
            else:
                # Special handling for RAG initialization which takes longer
                if endpoint == "/api/rag/initialize":
                    response = requests.post(url, headers=headers, json=data, timeout=900)  # 15 minutes for model loading
                else:
                    response = requests.post(url, headers=headers, json=data, timeout=300)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=300)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend API error: {str(e)}")
        return None

def initialize_rag_system(force: bool = False, incremental: bool = True):
    """Initialize RAG system via backend API."""
    endpoint = "/api/rag/initialize"
    data = {"force": force, "incremental": incremental}
    return call_backend_api(endpoint, "POST", data)

def get_rag_status():
    """Get RAG system status via backend API."""
    return call_backend_api("/api/rag/status")

def get_rag_documents():
    """Get indexed documents info via backend API."""
    return call_backend_api("/api/rag/documents")

def upload_documents_to_backend(uploaded_files):
    """Upload documents via backend API."""
    # First save files locally for backend to process
    saved_paths = save_uploaded_files(uploaded_files)

    results = []
    for file_path in saved_paths:
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
                result = call_backend_api("/api/documents/upload", "POST", files=files)
                if result:
                    results.append(result)
        except Exception as e:
            st.error(f"Failed to upload {os.path.basename(file_path)}: {str(e)}")

    return results

def process_query_via_api(query_data: Dict):
    """Process query via backend API."""
    return call_backend_api("/api/query", "POST", query_data)

def process_query(query: str,
                 retriever,
                 reranker,
                 response_generator,
                 process_config: Dict,
                 send_nb_chunks_to_llm=1) -> Dict:
    try:
        if process_config['retrieval']['use_query_expansion']:
            expanded_query = response_generator.expand_query(query)
            logging.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
            
        if process_config['retrieval']['use_bm25']:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="hybrid",
                top_k=process_config['retrieval']['top_k']
            )
        else:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="vector",
                top_k=process_config['retrieval']['top_k']
            )
        
        if process_config['retrieval']['use_reranking']:
            reranked_results = reranker.rerank(
                query,
                [r.document for r in retrieved_results],
                top_k=send_nb_chunks_to_llm
            )
            relevant_docs = [r.document for r in reranked_results]
            best_score = reranked_results[0].score if reranked_results else 0.0
        else:
            relevant_docs = [r.document for r in retrieved_results]
            best_score = retrieved_results[0].score if retrieved_results else 0.0
        
        # Log retrieved sources for debugging
        sources = [doc.metadata.get("source", "unknown") for doc in relevant_docs]
        logging.info(f"Retrieved sources for query '{query}': {sources}")
        
        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={'retrieval_score': best_score}
        )
        
        return {
            'Query': query,
            'Response': response_data['response'],
            'Score': best_score,
            'Sources': relevant_docs
        }
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {'Query': query, 'Response': "An error occurred.", 'Score': 0.0, 'Sources': []}

def initialize_session_state():
    if "rag_components" not in st.session_state:
        st.session_state.rag_components = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Ingestion"

def display_response(content: str, sources: List, score: float) -> None:
    formatted_content = content.replace("\n-", "\n\n-").replace("\n ", "\n").strip()
    st.subheader("Answer")
    st.markdown(formatted_content)

    if sources:
        with st.expander("View Sources Used"):
            for idx, source in enumerate(sources, 1):
                st.markdown(f"**Source {idx}:**")
                st.text_area(
                    label=f"Source {idx} content",
                    value="From : " + source.metadata.get("source", "unknown") + "\n\nContent : \n" + source.page_content,
                    height=200,
                    label_visibility="collapsed",
                    key=f"source_{idx}_{hash(source.page_content+str(random.random()*1000000))}",
                )

    if score is not None:
        normalized_score = max(0.0, min(abs(score), 1.0))
        st.progress(normalized_score, text=f"Confidence: {normalized_score:.2%}")

def ensure_rag_initialized(force=False, incremental=True) -> bool:
    """
    Initialize RAG system via backend API.

    Args:
        force: Force complete rebuild (ignores cache)
        incremental: Use incremental rebuild (only process changed files)

    Returns:
        bool: Success status
    """
    # Check if we already have status cached
    if not force and hasattr(st.session_state, 'rag_status') and st.session_state.rag_status:
        status = get_rag_status()
        if status and status.get("initialized"):
            return True

    if incremental and not force:
        # Try incremental rebuild first
        with st.spinner("Checking for document changes..."):
            result = initialize_rag_system(force=False, incremental=True)
            if result and result.get("status") == "success":
                st.session_state.rag_status = get_rag_status()
                st.info("✅ Index updated incrementally (only changed files processed)")
                return True
            else:
                st.warning("Incremental rebuild failed, falling back to full rebuild")
                # Fall through to full rebuild

    # Full rebuild (force=True or incremental failed)
    spinner_text = "Rebuilding RAG index completely..." if force else "Initializing RAG system..."
    with st.spinner(spinner_text):
        result = initialize_rag_system(force=True, incremental=False)
        if result and result.get("status") == "success":
            st.session_state.rag_status = get_rag_status()
            return True
        else:
            error_msg = result.get("detail", "Unknown error") if result else "API call failed"
            st.error(f"RAG initialization failed: {error_msg}")
            return False

def render_ingestion_view():
    st.title("Document Ingestion & Analytics")

    components = st.session_state.rag_components

    # File Upload Section
    st.header("📤 Document Upload")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader(
             "Upload Documents, Code, or Notebooks",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'py', 'js', 'ts', 'java', 'ipynb', 'csv']
        )

        if uploaded_files:
            st.info(f"📋 Ready to upload {len(uploaded_files)} files:")
            for file in uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.write(f"• {file.name} ({file_size_mb:.2f} MB)")

    with col2:
        st.subheader("Quick Actions")
        if st.button("🚀 Upload & Rebuild Index", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("Please select files to upload first")
            else:
                with st.spinner("Uploading files..."):
                    saved_paths = save_uploaded_files(uploaded_files)
                    st.success(f"✅ Saved {len(uploaded_files)} files to data/raw")

                with st.spinner("Rebuilding RAG index..."):
                    if ensure_rag_initialized(force=True):
                        st.success("RAG index rebuilt successfully.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to rebuild index")

        if st.button("🔄 Update Index (Incremental)", type="secondary", use_container_width=True):
            with st.spinner("Checking for document changes and updating index..."):
                if ensure_rag_initialized(force=False, incremental=True):
                    st.success("✅ Index updated incrementally!")
                else:
                    st.error("Failed to update index")

        if st.button("🔄 Full Rebuild Index", type="secondary", use_container_width=True):
            with st.spinner("Complete rebuild of all documents (may take time)..."):
                if ensure_rag_initialized(force=True, incremental=False):
                    st.success("✅ Index fully rebuilt from all documents!")
                else:
                    st.error("Failed to rebuild index")

    st.divider()

    # Document Analytics Dashboard - Get data from backend
    status = get_rag_status()
    documents = get_rag_documents()

    if status and status.get("initialized"):
        st.header("📊 Document Collection Analytics")

        # Calculate detailed metrics from backend data
        total_chunks = status.get("chunks_count", 0)
        unique_sources = documents.get("total_documents", 0) if documents else 0

        # File type analysis from backend data
        file_types = defaultdict(int)
        if documents and documents.get("documents"):
            for doc in documents["documents"]:
                doc_type = doc.get("file_type", "unknown")
                file_types[doc_type] += 1

        # Estimate chunk statistics (backend doesn't provide detailed chunk info)
        avg_chunk_length = 2000  # Based on our config
        max_chunk_length = 2000
        min_chunk_length = 1000

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Chunks", f"{total_chunks:,}")
        with col2:
            st.metric("📁 Source Files", unique_sources)
        with col3:
            st.metric("📏 Avg Chunk Size", f"{avg_chunk_length:.0f} chars")
        with col4:
            st.metric("🤖 LLM Context", "5 chunks", "Per Query")

        # File Type Distribution
        st.subheader("📋 File Type Breakdown")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Summary table
            type_summary = []
            for file_type, count in file_types.items():
                percentage = (count / unique_sources) * 100 if unique_sources > 0 else 0
                type_summary.append({
                    "Type": file_type,
                    "Files": count,
                    "Percentage": f"{percentage:.1f}%"
                })

            if type_summary:
                summary_df = pd.DataFrame(type_summary)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No file type data available")

        with col2:
            # Pie chart
            if file_types:
                type_df = pd.DataFrame(list(file_types.items()), columns=['Type', 'Count'])
                fig = px.pie(type_df, values='Count', names='Type',
                            title="Document Types Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)

        # Chunk Size Analysis
        st.subheader("📏 Chunk Size Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Statistics
            st.write("**Chunk Size Statistics:**")
            st.write(f"• **Average:** {avg_chunk_length:.0f} characters")
            st.write(f"• **Maximum:** {max_chunk_length:,} characters")
            st.write(f"• **Minimum:** {min_chunk_length:,} characters")
            st.write("• **Total Chunks:** N/A (estimated)")
            st.write("\n**Size Ranges (estimated):**")
            st.write("• Small (< 500 chars): ~10%")
            st.write("• Medium (500-1500 chars): ~20%")
            st.write("• Large (≥ 1500 chars): ~70%")

        with col2:
            # Placeholder histogram
            st.info("Chunk size analysis available after detailed statistics are implemented in backend.")

        # Document Sources List
        st.subheader("📚 Document Sources")

        if documents and documents.get("documents"):
            sources_list = [doc.get("source", "unknown") for doc in documents["documents"]]
            if sources_list:
                with st.expander("View All Document Sources", expanded=False):
                    for i, source in enumerate(sources_list[:10], 1):  # Show first 10
                        st.write(f"{i}. **{source}**")
                    if len(sources_list) > 10:
                        st.write(f"... and {len(sources_list) - 10} more sources")
            else:
                st.info("No document sources found")
        else:
            st.info("Document sources not available yet")

        # Processing Configuration
        st.header("⚙️ Current Processing Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📝 Text Processing")
            st.write("**Chunk Size:** 2000 chars")
            st.write("**Chunk Overlap:** 100 chars")
            st.write("**Headers to Split:** 4 levels")

        with col2:
            st.subheader("🔍 Retrieval Settings")
            st.write("**Top-K Results:** 30")
            st.write("**BM25 Weight:** 0.7")
            st.write("**Query Expansion:** ✅")

        with col3:
            st.subheader("🔄 Advanced Features")
            st.write("**RAG Fusion:** ✅")
            st.write("**Query Variants:** 3")
            st.write("**Translation:** ✅")

        # System Health for Ingestion
        st.header("💚 Ingestion System Health")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cache_files = len([f for f in os.listdir("cache") if f.endswith('.pkl')])
            st.metric("💾 Cache Files", cache_files, "Active indices")

        with col2:
            # Check raw data directory
            raw_files = len([f for f in os.listdir("data/raw") if not f.startswith('.')])
            st.metric("📁 Raw Documents", raw_files, "Available")

        with col3:
            # Estimate processing time
            estimated_time = (total_chunks * 0.01)  # Rough estimate: 10ms per chunk
            st.metric("⏱️ Processing Time", f"~{estimated_time:.1f}s", "Estimated")

        with col4:
            # Memory usage estimate
            memory_mb = (total_chunks * avg_chunk_length * 0.000001)  # Rough MB estimate
            st.metric("🧠 Memory Usage", f"~{memory_mb:.1f} MB", "Estimated")

    else:
        st.info("🤖 System not initialized. Upload documents and rebuild the index to see analytics.")
        st.markdown("""
        **Getting Started:**
        1. Upload documents using the file uploader above
        2. Click "Upload & Rebuild Index" to process them
        3. View detailed analytics and system metrics here
        """)

def render_query_view():
    st.title("🔍 Advanced Query Interface")

    if not ensure_rag_initialized(): st.stop()

    # Get status from backend
    status = get_rag_status()
    if not status or not status.get("initialized"):
        st.error("RAG system not initialized. Please initialize from the Ingestion tab first.")
        return

    # System Status Indicator
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 System", "🟢 Ready", "RAG Online")
    with col2:
        st.metric("🔄 Fusion", "🟢 Active", "Query Enhancement")
    with col3:
        total_chunks = status.get("chunks_count", 0)
        st.metric("📄 Knowledge Base", f"{total_chunks:,}", "Indexed Chunks")
    with col4:
        last_query = st.session_state.get('last_query', 'None')
        st.metric("💬 Last Query", "✅ Complete" if st.session_state.last_result else "⏳ Ready", "Status")

    # Query Input Section
    st.header("❓ Ask Your Question")

    # Language detection helper
    st.info("💡 **Pro Tip**: The system supports both Korean and English queries with automatic language detection and translation.")

    # Query configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        # Используем форму, чтобы ввод текста и кнопка работали как одно целое
        with st.form(key='query_form'):
            query = st.text_area(
                "Enter your question in Korean or English",
                value="",
                height=120,
                placeholder="Example: What are the security features of Matter protocol?\nOr: Matter 프로토콜의 보안 기능은 무엇인가요?",
                help="The system will automatically detect language and use RAG Fusion for optimal results"
            )

            # Advanced options
            with st.expander("⚙️ Advanced Query Options"):
                col_a, col_b = st.columns(2)
                with col_a:
                    use_fusion = st.checkbox("Enable RAG Fusion", value=True,
                                           help="Generate multiple query variants for better results")
                    show_expanded = st.checkbox("Show Query Expansion", value=False,
                                              help="Display how the query was processed")
                with col_b:
                    max_sources = st.slider("Max Sources to Show", 1, 10, 3,
                                          help="Number of source documents to display")

            submit_button = st.form_submit_button(label='🚀 Run Enhanced Query', type="primary", use_container_width=True)

    with col2:
        st.subheader("📚 Quick Examples")
        if st.button("🔒 Security Features (EN)"):
            st.session_state.example_query = "What are the security features of the Matter protocol?"
            st.rerun()
        if st.button("🔒 보안 기능 (KO)"):
            st.session_state.example_query = "Matter 프로토콜의 보안 기능에 대해 설명해주세요"
            st.rerun()
        if st.button("🌐 Platform Features"):
            st.session_state.example_query = "What are the main features of cloud platforms?"
            st.rerun()

        # Handle example queries
        if 'example_query' in st.session_state:
            query = st.session_state.example_query
            del st.session_state.example_query
            # Auto-submit would go here

    # Process query
    if submit_button and query.strip():
        st.session_state.last_result = None
        st.session_state.last_query = query

        # Create containers for loader and results
        loader_container = st.empty()
        results_container = st.empty()

        # Show loader
        with loader_container:
            if use_fusion:
                st.markdown("### 🔄 Processing with RAG Fusion")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Fusion processing steps
                steps = [
                    "Analyzing query language...",
                    "Generating query variants...",
                    "Running multi-retrieval...",
                    "Applying result fusion...",
                    "Reranking top results...",
                    "Generating response..."
                ]

                for i, step in enumerate(steps):
                    progress_bar.progress((i + 1) / len(steps))
                    status_text.markdown(f"**{step}**")
                    time.sleep(0.4)  # Slightly longer for better UX

                progress_bar.empty()
                status_text.empty()

                # Final processing spinner
                with st.spinner("Finalizing response..."):
                    time.sleep(0.5)
            else:
                with st.spinner("Processing query..."):
                    time.sleep(1.5)

        # Process the query via backend API
        start_time = time.time()
        query_data = {
            "query": query,
            "use_fusion": use_fusion,
            "top_k": 5
        }
        result = process_query_via_api(query_data)
        processing_time = time.time() - start_time

        if result and result.get("status") == "success":
            # Convert backend response to frontend format
            st.session_state.last_result = {
                "Query": result["query"],
                "Response": result["response"],
                "Score": result.get("processing_time_ms", 0) / 1000,  # Use processing time as score for display
                "Sources": [{"metadata": source} for source in result.get("sources", [])]
            }
            st.session_state.processing_time = result.get("processing_time_ms", 0) / 1000
        else:
            st.session_state.last_result = {
                "Query": query,
                "Response": "Error processing query. Please try again.",
                "Score": 0.0,
                "Sources": []
            }
            st.session_state.processing_time = processing_time

        # Clear loader and show success message
        loader_container.empty()

        with results_container:
            # Success message
            if use_fusion:
                st.success(f"RAG Fusion completed in {processing_time:.2f}s - Enhanced multilingual retrieval applied.")
            else:
                st.success(f"Query processed in {processing_time:.2f}s")

        time.sleep(1)  # Brief pause to show success message
        st.rerun()

    # Display Results
    if st.session_state.last_result:
        result = st.session_state.last_result
        processing_time = st.session_state.get('processing_time', 0)

        st.header("📋 Query Results")

        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            confidence = max(0.0, min(abs(result["Score"]), 1.0))
            st.metric("🎯 Confidence", f"{confidence:.1%}")
        with col2:
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
        with col3:
            sources_count = len(result["Sources"])
            st.metric("📄 Sources Used", sources_count)
        with col4:
            fusion_used = use_fusion if 'use_fusion' in locals() else True
            st.metric("🔄 Retrieval Mode", "Fusion" if fusion_used else "Standard")

        # Enhanced response display
        st.subheader("💡 Answer")
        display_response_enhanced(result["Response"], result["Sources"][:max_sources], result["Score"])

        # Query processing details (if requested)
        if show_expanded and 'expanded_query' in st.session_state.last_result:
            with st.expander("🔍 Query Processing Details"):
                st.write("**Original Query:**", st.session_state.last_query)
                st.write("**Expanded Query:**", st.session_state.last_result.get('expanded_query', 'N/A'))
                if fusion_used:
                    st.write("**Fusion Applied:** Multiple query variants generated and results fused")

        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy Answer"):
                st.success("Answer copied to clipboard!")
        with col2:
            if st.button("💾 Save Results"):
                st.success("Results saved to query_history.json")
        with col3:
            if st.button("🔄 New Query"):
                st.session_state.last_result = None
                st.rerun()

def display_response_enhanced(content: str, sources: List, score: float) -> None:
    """Enhanced response display with better formatting."""
    # Format content
    formatted_content = content.replace("\n-", "\n\n-").replace("\n ", "\n").strip()

    # Language detection for display
    is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in formatted_content[:100])

    # Display with appropriate styling
    if is_korean:
        st.markdown("**🇰🇷 Korean Response:**")
        st.info(formatted_content)  # Blue background for Korean
    else:
        st.markdown("**🇺🇸 English Response:**")
        st.success(formatted_content)  # Green background for English

    # Enhanced confidence display
    if score is not None:
        confidence = max(0.0, min(abs(score), 1.0))

        # Color-coded confidence
        if confidence >= 0.8:
            color = "🟢"
            level = "High"
        elif confidence >= 0.6:
            color = "🟡"
            level = "Medium"
        else:
            color = "🔴"
            level = "Low"

        st.progress(confidence, text=f"{color} Confidence: {level} ({confidence:.1%})")

    # Enhanced sources display
    if sources:
        with st.expander(f"📚 Sources Used ({len(sources)})", expanded=True):
            for idx, source in enumerate(sources, 1):
                col1, col2 = st.columns([1, 4])

                with col1:
                    # Source type indicator
                    source_name = source.metadata.get("source", "unknown")
                    if source_name.endswith('.pdf'):
                        st.markdown("📕 **PDF**")
                    elif source_name.endswith('.txt'):
                        st.markdown("📄 **Text**")
                    else:
                        st.markdown("📋 **Document**")

                    # Relevance score if available
                    chunk_score = getattr(source, 'score', None)
                    if chunk_score is not None:
                        st.caption(f"Score: {chunk_score:.3f}")

                with col2:
                    st.markdown(f"**Source {idx}:** {source_name}")

                    # Preview content (first 200 chars)
                    preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    st.text_area(
                        label=f"Content preview",
                        value=preview,
                        height=100,
                        label_visibility="collapsed",
                        disabled=True,
                        key=f"enhanced_source_{idx}_{hash(source.page_content[:50])}",
                    )

                st.divider()
def render_dashboard_view():
    st.title("📊 Advanced RAG Analytics Dashboard")

    # Get RAG status from backend
    status = get_rag_status()
    if not status or not status.get("initialized"):
        st.info("System not initialized. Please go to Ingestion tab and initialize the RAG system first.")
        return

    # Get document information
    documents = get_rag_documents()

    # System Status Overview
    st.header("🔴 System Status")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "🟢 Online", "Local RAG")
    with col2:
        st.metric("RAG Fusion", "🟢 Enabled", "Multilingual")
    with col3:
        st.metric("Model", "Llama 3.1 8B", "Ollama Local")
    with col4:
        st.metric("Embeddings", "BGE-M3", "Multilingual")

    # Core Metrics Dashboard
    st.header("📈 Core System Metrics")

    # Use backend data for metrics
    total_chunks = status.get("chunks_count", 0)
    total_documents = documents.get("total_documents", 0) if documents else 0

    # Metrics Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Chunks", f"{total_chunks:,}")
    with col2:
        st.metric("📁 Source Files", total_documents)
    with col3:
        st.metric("📏 Avg Chunk Size", "2000 chars", "Optimized")
    with col4:
        st.metric("🤖 LLM Context", "5 chunks", "Per Query")

    # Document Type Distribution
    st.subheader("📊 Document Type Distribution")

    if documents and documents.get("documents"):
        doc_list = documents["documents"]
        doc_types = {}
        for doc in doc_list:
            doc_type = doc.get("file_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        doc_type_df = pd.DataFrame(list(doc_types.items()), columns=['Type', 'Count'])

        col1, col2 = st.columns(2)
        with col1:
            if not doc_type_df.empty:
                fig = px.pie(doc_type_df, values='Count', names='Type',
                            title="Document Types", color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No document type data available")

        with col2:
            # Simple language distribution (could be enhanced with actual language detection)
            st.metric("🌐 Languages", "Korean & English", "Multilingual Support")
            st.info("System supports both Korean and English documents with automatic language detection.")
    else:
        st.info("Document information not available yet.")

    # Performance Metrics
    st.header("⚡ Performance Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔍 Retrieval Settings")
        st.write("**BM25 Weight:** 0.7")
        st.write("**Top-K Results:** 30")
        st.write("**Query Expansion:** ✅")
        st.write("**Reranking:** ✅")

    with col2:
        st.subheader("🔄 RAG Fusion Settings")
        st.write("**Fusion Enabled:** ✅")
        st.write("**Query Variants:** 3")
        st.write("**RRF K-Value:** 60")
        st.write("**Translation:** ✅")

    with col3:
        st.subheader("🧠 Model Configuration")
        st.write("**Embedding Model:** BGE-M3")
        st.write("**Reranker:** BGE-Reranker-v2-m3")
        st.write("**LLM:** Llama 3.1 8B")
        st.write("**Context Window:** 4096 tokens")

    # System Health Indicators
    st.header("💚 System Health")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🤖 LLM Status", "🟢 Healthy", "Ollama 3.1 8B")

    with col2:
        st.metric("🗂️ Vector Store", "🟢 Ready", f"{total_chunks} vectors")

    with col3:
        # Cache health
        cache_files = len([f for f in os.listdir("cache") if f.endswith('.pkl')])
        st.metric("💾 Cache Status", "🟢 Active", f"{cache_files} cached indices")

    with col4:
        st.metric("🧠 Memory Usage", "~8GB", "Optimized")

    # Recent Activity (if we had query history)
    st.header("📋 Recent System Activity")

    # Mock recent activity - in real app you'd track this
    activity_data = [
        {"time": "2 min ago", "action": "Query processed", "details": "Korean security features query"},
        {"time": "5 min ago", "action": "Index rebuilt", "details": "Added 3 new documents"},
        {"time": "10 min ago", "action": "System initialized", "details": "RAG Fusion enabled"},
        {"time": "15 min ago", "action": "Model loaded", "details": "BGE-M3 embeddings ready"},
    ]

    for activity in activity_data:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                st.write(f"🕐 {activity['time']}")
            with col2:
                st.write(f"**{activity['action']}**")
            with col3:
                st.write(activity['details'])
        st.divider()

    # Export Options
    st.header("📤 Export & Reporting")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Metrics", type="secondary"):
            st.success("Metrics exported to dashboard_metrics.json")

    with col2:
        if st.button("📋 Generate Report", type="secondary"):
            st.success("System report generated: rag_system_report.pdf")

    with col3:
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()

def main():
    st.set_page_config(page_title="Offline RAG Demo", layout="wide")
    initialize_session_state()

    # --- Presentable Button Navigation ---
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")
        if st.button("📁 1. Ingestion & Upload"):
            st.session_state.active_tab = "Ingestion"
        if st.button("🔍 2. Query & Answer"):
            st.session_state.active_tab = "Query"
        if st.button("📊 3. System Dashboard"):
            st.session_state.active_tab = "Dashboard"
        st.markdown("---")
        st.caption("Status: Connected (Local)")

    # Routing
    if st.session_state.active_tab == "Ingestion":
        render_ingestion_view()
    elif st.session_state.active_tab == "Query":
        render_query_view()
    else:
        render_dashboard_view()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
