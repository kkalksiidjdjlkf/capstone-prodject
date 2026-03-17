import argparse
import logging
from typing import Dict
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

from initialize_rag import RAGInitializer, RAGComponents
from src.utils.helpers import save_results

def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute evaluation metrics for predictions vs references."""
    bleu_scores = []
    semantic_scores = []
    
    # Load model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for pred, ref in zip(predictions, references):
        # BLEU score
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        bleu = sentence_bleu(ref_tokens, pred_tokens)
        bleu_scores.append(bleu)
        
        # Semantic similarity
        pred_emb = model.encode(pred, convert_to_tensor=True)
        ref_emb = model.encode(ref, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(pred_emb, ref_emb).item()
        semantic_scores.append(sim)
    
    return {
        'avg_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        'avg_semantic_sim': sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0,
        'bleu_scores': bleu_scores,
        'semantic_scores': semantic_scores
    }

def process_query(query: str,
                 retriever,
                 reranker,
                 response_generator,
                 process_config: Dict) -> Dict:
    """
    Process a single query through the complete RAG pipeline.
    
    This function orchestrates the query processing workflow:
    1. Optional query expansion
    2. Document retrieval
    3. Optional result reranking
    4. Response generation
    
    Args:
        query (str): The user's query
        retriever: Document retrieval component
        reranker: Result reranking component
        response_generator: Response generation component
        process_config (Dict): Processing configuration
        send_nb_chunks_to_llm (int): Number of chunks to send to LLM
        
    Returns:
        Dict: Processing results containing:
            - Query: Original query
            - Response: Generated response
            - Score: Best retrieval/reranking score
            
    Note:
        The function handles errors gracefully, returning an error message
        in the response if any step fails.
    """
    try:
        # Expand query if configured
        if process_config['retrieval']['use_query_expansion']:
            expanded_query = response_generator.expand_query(query)
            logging.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
            
        # Retrieve relevant documents using expanded or original query
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
        logging.info(f"Retrieved {len(retrieved_results)} documents")
        
        # Apply reranking if configured
        if process_config['retrieval']['use_reranking']:
            reranked_results = reranker.rerank(
                query,
                [r.document for r in retrieved_results],
                top_k=process_config['retrieval']['send_nb_chunks_to_llm']
            )
            relevant_docs = [r.document for r in reranked_results]
            best_score = reranked_results[0].score if reranked_results else 0.0
            logging.info(f"Reranked results. Best score: {best_score}")
        else:
            relevant_docs = [r.document for r in retrieved_results]
            best_score = retrieved_results[0].score if retrieved_results else 0.0
            logging.info(f"Using retrieval scores. Best score: {best_score}")
        
        # Generate final response using selected documents
        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={'retrieval_score': best_score}
        )
        
        return {
            'Query': query,
            'Response': response_data['response'].strip()[:-4],
        }
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {
            'Query': query,
            'Response': "An error occurred processing your query.",
        }

def process_test_queries(components: RAGComponents):
    """Process all test queries using initialized components."""
    logging.info("Initializing retriever...")
    components.retriever.initialize(components.original_chunks)
    
    logging.info("Processing test queries...")
    for idx, row in tqdm(components.test_df.iterrows(), total=len(components.test_df)):
        # Skip already processed queries
        if (components.test_results is not None and 
            row['trustii_id'] in components.test_results['trustii_id'].values):
            continue
            
        # Handle invalid queries
        if pd.isna(row['Query']):
            results = {
                'trustii_id': row['trustii_id'],
                'Query': row['Query'],
                'Response': pd.NA
            }
        else:
            results = process_query(
                row['Query'],
                components.retriever,
                components.reranker,
                components.response_generator,
                components.process_config
            )
            results['trustii_id'] = row['trustii_id']
            
        save_results(results, components.init_config['files']['test_output'], is_test=True)

def main(init_config_path: str, process_config_path: str):
    """Main execution function."""
    try:
        # Initialize system with both configs
        initializer = RAGInitializer(init_config_path, process_config_path)
        components = initializer.initialize()
        
        # Process queries
        process_test_queries(components)
        
        # Load results and compute metrics
        results_df = pd.read_csv(components.init_config['files']['test_output'])
        train_df = pd.read_csv(components.init_config['files']['train_data'])
        
        # Simple evaluation: match queries and compute metrics
        predictions = []
        references = []
        for _, row in results_df.iterrows():
            query = row['Query']
            # Find closest match in train_df
            closest = train_df[train_df['Query'].str.contains(query, case=False, na=False)]
            if not closest.empty:
                pred = row['Response']
                ref = closest.iloc[0]['Response']
                predictions.append(pred)
                references.append(ref)
        
        if predictions and references:
            metrics = compute_metrics(predictions, references)
            logging.info(f"Evaluation Metrics: BLEU={metrics['avg_bleu']:.4f}, Semantic Sim={metrics['avg_semantic_sim']:.4f}")
            # Save metrics
            metrics_df = pd.DataFrame({
                'query': results_df['Query'][:len(predictions)],
                'prediction': predictions,
                'reference': references,
                'bleu': metrics['bleu_scores'],
                'semantic_sim': metrics['semantic_scores']
            })
            metrics_df.to_csv('data/raw/evaluation_metrics.csv', index=False)
        else:
            logging.warning("No matching queries found for evaluation.")
        
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Query Processing')
    parser.add_argument('--init-config', 
                       type=str, 
                       default='config/init_config.yaml',
                       help='Path to initialization configuration file')
    parser.add_argument('--process-config', 
                       type=str, 
                       default='config/process_config.yaml',
                       help='Path to processing configuration file')
    args = parser.parse_args()
    
    main(args.init_config, args.process_config)