"""
Query processing module for the RAG (Retrieval-Augmented Generation) system.
Updated to support hybrid retrieval and persistent vector store loading.
"""
import traceback
import argparse
import logging
import os
from typing import Dict

from initialize_rag import RAGInitializer

def process_query(query: str,
                 retriever,
                 reranker,
                 response_generator,
                 process_config: Dict,
                 send_nb_chunks_to_llm=1) -> Dict:
    """
    Process a single query through the complete RAG pipeline.
    """
    try:
        # 1. Expand query if configured (Translation/Expansion)
        if process_config['retrieval']['use_query_expansion']:
            expanded_query = response_generator.expand_query(query)
            logging.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
            
        # 2. Retrieve relevant documents using Fusion or Hybrid Method
        if process_config.get('fusion', {}).get('enabled', False):
            # Use RAG Fusion with query variants
            retrieved_results = retriever.retrieve_with_fusion(
                expanded_query,
                top_k=process_config['retrieval']['top_k']
            )
            logging.info(f"Retrieved {len(retrieved_results)} documents using RAG Fusion")
        else:
            # Use traditional hybrid retrieval
            method = "hybrid" if process_config['retrieval']['use_bm25'] else "vector"
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method=method,
                top_k=process_config['retrieval']['top_k']
            )
            logging.info(f"Retrieved {len(retrieved_results)} documents using {method} search")
        
        # 3. Apply reranking if configured
        if process_config['retrieval']['use_reranking'] and reranker:
            # Превращаем RetrievalResult обратно в список Document для реранкера
            docs_to_rerank = [r.document for r in retrieved_results]
            reranked_results = reranker.rerank(
                query,
                docs_to_rerank,
                top_k=send_nb_chunks_to_llm
            )
            relevant_docs = [r.document for r in reranked_results]
            best_score = reranked_results[0].score if reranked_results else 0.0
            logging.info(f"Reranked results. Best score: {best_score}")
        else:
            relevant_docs = [r.document for r in retrieved_results[:send_nb_chunks_to_llm]]
            best_score = retrieved_results[0].score if retrieved_results else 0.0
            logging.info(f"Using retrieval scores. Best score: {best_score}")
        
        # 4. Generate final response (with Korean context support)
        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={'retrieval_score': best_score}
        )
        
        return {
            'Query': query,
            'Response': response_data['response'],
            'Score': best_score
        }
        
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error processing query: {str(e)}")
        return {
            'Query': query,
            'Response': "An error occurred processing your query.",
            'Score': 0.0
        }

def main(init_config_path: str, process_config_path: str, query: str):
    """
    Main execution function optimized for speed via hybrid initialization.
    """
    try:
        # 1. Initialize system components
        initializer = RAGInitializer(init_config_path, process_config_path)
        components = initializer.initialize()
        
        # 2. Unified Initialization (Handles both BM25 and Vector Store)
        # Внутри HybridRetriever.initialize уже заложена логика:
        # если кэш FAISS есть - загрузить, если нет - создать.
        logging.info("--- Initializing Hybrid Retrieval System (BGE-M3 + BM25) ---")
        
        # Мы вызываем общий метод инициализации, который подготовит обе базы
        components.retriever.initialize(components.original_chunks)

        # 3. Process query
        result = process_query(
            query=query,
            retriever=components.retriever,
            reranker=components.reranker,
            response_generator=components.response_generator,
            process_config=components.process_config,
            send_nb_chunks_to_llm=components.process_config['retrieval']['send_nb_chunks_to_llm']
        )
        
        print("\n" + "="*50)
        print("QUERY:", result["Query"])
        print("RESPONSE:", result["Response"])
        print("SCORE (Relevance):", f"{result['Score']:.4f}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Query Processing System')
    parser.add_argument('--init-config', type=str, default='config/init_config.yaml')
    parser.add_argument('--process-config', type=str, default='config/process_config.yaml')
    parser.add_argument('--query', type=str, required=True)
    
    args = parser.parse_args()
    main(args.init_config, args.process_config, args.query)
