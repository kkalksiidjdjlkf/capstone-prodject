from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List, Dict, Optional
import logging
from queue import Queue
import threading
import time

try:
    from langdetect import detect
except ImportError:
    def detect(text):
        """Fallback language detection if langdetect not available."""
        if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
            return 'ko'  # Korean
        return 'en'  # Default to English

class ResponseGenerator:

    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe management
        self.llm_queue = Queue()
        self.llm_lock = threading.Lock()
        
        # FIX 1: Explicitly set model name as a string to avoid Pydantic errors.
        # FIX 2: num_ctx expanded to 4096 to handle larger context from Matter specs.
        # FIX 3: num_predict set to 1024 to ensure the response finishes completely.
        self.llm = OllamaLLM(
            model="llama3.1:8b",
            temperature=0,      # Strict deterministic output for technical data
            num_ctx=4096,       # Input memory size (prevents overflow)
            num_predict=1024,    # Output length (prevents 'Internet Proto...' cutoff)
            base_url="http://host.docker.internal:11434"
        )
        
        # Validate Ollama setup
        self._validate_ollama()

    def _validate_ollama(self):
        """Validate Ollama connectivity and model availability."""
        try:
            # Simple test invoke to check if Ollama is running and model is loaded
            test_response = self.llm.invoke("Hello")
            if test_response:
                self.logger.info("Ollama validation successful: Model ready.")
            else:
                self.logger.warning("Ollama validation: No response from model.")
        except Exception as e:
            self.logger.error(f"Ollama validation failed: {str(e)}. Ensure Ollama is running and 'llama3.1:8b' is pulled.")
            raise RuntimeError("Ollama setup error. Check Ollama service and model availability.")

    def _invoke_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Invoke LLM with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(prompt)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"LLM invoke failed (attempt {attempt + 1}): {str(e)}. Retrying in {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"LLM invoke failed after {max_retries} attempts: {str(e)}")
                    raise

    def generate_answer(self,
                       query: str,
                       relevant_chunks: List[Document],
                       metadata: Optional[Dict] = None) -> Dict:
        try:
            # Detect query language
            try:
                query_lang = detect(query)
                self.logger.info(f"Detected query language: {query_lang} for query: {query[:50]}...")
            except Exception as e:
                query_lang = 'en'  # Default to English
                self.logger.warning(f"Language detection failed: {e}, defaulting to English")

            # Filter chunks by language if possible
            filtered_chunks = []
            english_chunks = []
            korean_chunks = []

            for chunk in relevant_chunks:
                try:
                    chunk_lang = detect(chunk.page_content[:500])  # Check first 500 chars
                    if chunk_lang == 'en':
                        english_chunks.append(chunk)
                    elif chunk_lang == 'ko':
                        korean_chunks.append(chunk)
                except:
                    # If detection fails, include in both for now
                    english_chunks.append(chunk)
                    korean_chunks.append(chunk)

            # Prioritize chunks in the same language as query
            if query_lang == 'ko' and korean_chunks:
                filtered_chunks = korean_chunks
                self.logger.info(f"Using {len(korean_chunks)} Korean chunks for Korean query")
            elif query_lang == 'en' and english_chunks:
                filtered_chunks = english_chunks
                self.logger.info(f"Using {len(english_chunks)} English chunks for English query")
            else:
                # Fallback to all chunks if no language match
                filtered_chunks = relevant_chunks
                self.logger.info(f"No language-matched chunks found, using all {len(relevant_chunks)} chunks")
            
            # Handle meta-queries about documents
            query_lower = query.lower()
            if ("description" in query_lower or "describe" in query_lower) and ("document" in query_lower or "doc" in query_lower or "file" in query_lower):
                sources = list(set([c.metadata.get("source", "unknown") for c in filtered_chunks if c.metadata.get("source")]))
                if sources:
                    response = f"Indexed documents: {', '.join(sources)}. Query specific topics for details."
                else:
                    response = "No documents are currently indexed."
                return {
                    'response': response,
                    'source_documents': [],
                    'metadata': metadata or {}
                }
            
            # Treat the highest-ranked chunk as the primary evidence source.
            primary_chunk = filtered_chunks[0] if filtered_chunks else None
            context = self._prepare_context(filtered_chunks)
            response = self._generate_response(query, context)
            
            return {
                'response': response,
                'source_documents': [doc.metadata for doc in filtered_chunks],
                'metadata': {
                    **(metadata or {}),
                    "primary_source": primary_chunk.metadata if primary_chunk else None,
                },
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'response': "An error occurred. Please try again. / 오류가 발생했습니다. 다시 시도해 주세요.",
                'error': str(e),
                'metadata': metadata or {}
            }

    def _prepare_context(self, chunks: List[Document]) -> str:
        """Formats chunks. Note: Limit to top 5-10 chunks in your main script to save memory."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Document Chunk {i} ---\n{chunk.page_content}")
        return "\n\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> str:
        """
        Refined KREPS prompt.
        Removed the hardcoded Korean 'error' sentence from instructions to stop language leakage.
        """
        
        # Detect query language for prompt customization
        try:
            query_lang = detect(query)
        except:
            query_lang = 'en'

        # Language-specific instructions
        if query_lang == 'ko':
            language_instructions = """
            CRITICAL: The user asked in KOREAN, so you MUST respond ONLY in KOREAN.
            - Use 존댓말 (honorific speech) throughout your response
            - Do NOT include ANY English words or phrases
            - If you cannot answer in Korean context, say: "제공된 맥락에서 관련 정보를 찾을 수 없습니다."
            """
            no_answer_text = "제공된 맥락에서 관련 정보를 찾을 수 없습니다."
        else:
            language_instructions = """
            CRITICAL: The user asked in ENGLISH, so you MUST respond ONLY in ENGLISH.
            - Do NOT include ANY Korean words or phrases (except proper nouns if necessary)
            - If you cannot answer from context, say: "I cannot find the relevant information in the provided context."
            """
            no_answer_text = "I cannot find the relevant information in the provided context."

        # Updated prompt with stronger language enforcement
        system_prompt = f"""
        You are a highly precise technical assistant for KREPS (Korea Renewable Energy and Power Solutions).
        Your task is to analyze the provided documentation and answer the user's question.

        {language_instructions}

        STRICT RULES:
        1. CONTENT LIMITS:
           - Use ONLY the provided Context Information below to answer.
           - If the context does not contain relevant information for the question, respond with exactly: "{no_answer_text}"
           - DO NOT use any external knowledge or information you were trained on.
           - DO NOT answer questions that are not covered by the provided context.

        2. ANSWER STRUCTURE:
           - Be concise but comprehensive.
           - Structure answers with clear sections if needed.
           - Use exact phrases from the context when possible.
           - Do NOT mention document names, chunk numbers, or sources in your response.

        3. LANGUAGE COMPLIANCE:
           - Follow the language instructions above WITHOUT EXCEPTION.
           - Match the query language exactly.

        Context Information:
        {{context}}

        User Question: {{query}}

        Answer (in the same language as the question):
        """

        prompt = ChatPromptTemplate.from_template(system_prompt)
        formatted_prompt = prompt.format(query=query, context=context)
        
        return self._invoke_with_retry(formatted_prompt)

    def expand_query(self, query: str) -> str:
        """Optimizes queries while respecting technical acronyms and adding cross-lingual support."""
        expansion_prompt = """
        Analyze the query for technical document retrieval.
        - If the query is in English, also provide a Korean translation for better matching.
        - If the query is in Korean, also provide an English translation.
        - Keep acronyms (HMI, DAC, PASE, CASE, KT Cloud) unchanged.
        - Return ONLY the optimized query (original + translations if applicable).

        Query: {query}
        Output:"""

        prompt = ChatPromptTemplate.from_template(expansion_prompt)
        return self._invoke_with_retry(prompt.format(query=query))

    def generate_query_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate multiple query variants for RAG Fusion."""
        variants = [query]  # Always include original

        try:
            # Generate expanded/translated version
            expanded = self.expand_query(query)
            if expanded and expanded != query and "Optimized Query:" in expanded:
                # Parse the structured output
                lines = expanded.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('-') and 'Optimized Query:' not in line:
                        clean_variant = line.strip()
                        if clean_variant and clean_variant != query:
                            variants.append(clean_variant)
                            break  # Just take the first translation
            elif expanded and expanded != query:
                variants.append(expanded)

        except Exception as e:
            self.logger.warning(f"Query variant generation failed: {e}")

        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            clean_variant = variant.strip()
            if clean_variant and clean_variant not in seen:
                unique_variants.append(clean_variant)
                seen.add(clean_variant)

        return unique_variants[:num_variants]
