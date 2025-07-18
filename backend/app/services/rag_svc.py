# backend/app/services/rag_svc.py

import json 
from typing import List, Dict, Any, Optional, Tuple
import time 
import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.documents import Document as LangchainDocument 
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Local application imports
from backend.app import (
    OPENROUTER_API_BASE, 
    OPENROUTER_API_KEY, 
    DEFAULT_LLM_MODEL, 
    DEFAULT_LLM_PROVIDER, 
    PROJECT_NAME, 
    USE_OLLAMA, 
    OLLAMA_CHAT_MODEL, 
    OLLAMA_BASE_URL,
    GOOGLE_MODEL_NAME,
    GOOGLE_API_KEY,
    CROSS_ENCODER_MODEL
)
from backend.app.services.vstore_svc import VectorStoreService
from backend.app.types.response_format import RAGResponse, SVGResponseFormat, DynamicPrompt
from backend.app.services.system_message import SVG_PROMPT, SVG_GENERATION_PROMPT

# For Cross-Encoder Reranking
from sentence_transformers import CrossEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """
    Service to handle Retrieval Augmented Generation:
    - Retrieves relevant documents from the vector store (using MMR).
    - Reranks the retrieved documents using a Cross-Encoder.
    - Interacts with an LLM to generate answers and identify themes based on context.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        
        # Initialize Cross-Encoder for reranking
        try:
            self.reranker_model_name = CROSS_ENCODER_MODEL 
            self.reranker = CrossEncoder(self.reranker_model_name, device='cpu') 
            logger.info(f"RAGService initialized with Cross-Encoder: {self.reranker_model_name}")
        except Exception as e:
            logger.error(f"Error initializing CrossEncoder: {e}. Reranking will be skipped.")
            self.reranker = None
        
        # Initialize LLM based on configuration
        self._initialize_llm()
        
        # Set up JSON output parser
        self.json_parser = JsonOutputParser(pydantic_object=RAGResponse)

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration"""
        try:
            if USE_OLLAMA:
                self._llm = ChatOllama(
                    model=OLLAMA_CHAT_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.4,
                )
                self.__llm = ChatGoogleGenerativeAI(
                    model=GOOGLE_MODEL_NAME,
                    api_key=GOOGLE_API_KEY
                )
                # Set up structured output for RAG responses with Ollama
                self.structured_llm = self._llm.with_structured_output(RAGResponse)
                
                # Set up memory and conversation chain for simple responses
                self.memory = ConversationSummaryMemory(
                    llm=self._llm, 
                    return_messages=True,
                    max_token_limit=5000
                )
                self.conversation_chain = ConversationChain(
                    llm=self._llm, 
                    memory=self.memory,
                    verbose=False
                )
                
                # Set up chains for SVG generation
                self.prompt_chain = SVG_GENERATION_PROMPT | self._llm.with_structured_output(DynamicPrompt)
                self.svg_chain = SVG_PROMPT | self.__llm.with_structured_output(SVGResponseFormat)
                
                logger.info(f"Initialized Ollama LLM: {OLLAMA_CHAT_MODEL}")
                
            else:
                if not OPENROUTER_API_KEY:
                    raise ValueError("OpenRouter API key not configured")
                
                self._llm = ChatOpenAI(
                    model=DEFAULT_LLM_MODEL,
                    openai_api_key=OPENROUTER_API_KEY,
                    openai_api_base=OPENROUTER_API_BASE,
                    temperature=0.1,
                    max_tokens=3500,
                    default_headers={
                        "HTTP-Referer": PROJECT_NAME,
                        "X-Title": PROJECT_NAME
                    }
                )

                self.__llm = ChatGoogleGenerativeAI(
                    model=GOOGLE_MODEL_NAME,
                    api_key=GOOGLE_API_KEY
                )
                
                # Set up structured output for RAG responses with OpenRouter
                self.structured_llm = self._llm.with_structured_output(RAGResponse)
                
                # Set up memory and conversation chain for simple responses
                self.memory = ConversationSummaryMemory(
                    llm=self._llm, 
                    return_messages=True,
                    max_token_limit=5000
                )
                self.conversation_chain = ConversationChain(
                    llm=self._llm, 
                    memory=self.memory,
                    verbose=False
                )
                
                # Set up chains for SVG generation
                self.prompt_chain = SVG_GENERATION_PROMPT | self._llm.with_structured_output(DynamicPrompt)
                self.svg_chain = SVG_PROMPT | self.__llm.with_structured_output(SVGResponseFormat)
            
        except Exception as e:
            logger.error(f"Unable to configure LLM: {e}")
            self._llm = None
            self.structured_llm = None
            self.conversation_chain = None

    def _call_llm_structured(self, prompt: str) -> Optional[RAGResponse]:
        """Call LLM with structured output parsing for RAGResponse format"""
        if not self.structured_llm:
            logger.error("Structured LLM not initialized")
            return None
            
        try:
            logger.info("Calling LLM for structured response...")
            
            if USE_OLLAMA:
                # For Ollama, we need to be more explicit about structured output
                try:
                    # First attempt with structured output
                    response = self.structured_llm.invoke(prompt)
                    if isinstance(response, RAGResponse):
                        return response
                    
                    # If that fails, try manual parsing
                    if isinstance(response, str):
                        parsed_response = self.json_parser.parse(response)
                        return RAGResponse(**parsed_response)
                    
                except Exception as ollama_error:
                    logger.warning(f"Ollama structured output failed: {ollama_error}, trying fallback")
                    
                    # Fallback: use conversation chain with format instructions
                    format_instructions = self.json_parser.get_format_instructions()
                    full_prompt = f"{prompt}\n\n{format_instructions}"
                    
                    response = self.conversation_chain.invoke({"input": full_prompt})
                    response_text = response.get("response", str(response))
                    
                    try:
                        parsed_response = self.json_parser.parse(response_text)
                        return RAGResponse(**parsed_response)
                    except Exception as parse_error:
                        logger.error(f"Failed to parse Ollama response: {parse_error}")
                        # Return basic response structure
                        return RAGResponse(
                            answer=response_text,
                            identified_themes=[],
                            references=[],
                            context_assessment="Unable to parse structured response",
                            limitations="Response parsing failed"
                        )
            else:
                # For OpenRouter, use structured output directly
                response = self.structured_llm.invoke(prompt)
                return response
                
        except Exception as e:
            logger.error(f"Error calling structured LLM: {e}")
            return None

    def _call_llm_simple(self, prompt: str) -> Optional[str]:
        """Call LLM for simple text response using conversation chain"""
        if not self.conversation_chain:
            logger.error("Conversation chain not initialized")
            return None
            
        try:
            logger.info("Calling LLM for simple response using conversation chain...")
            
            # Use conversation chain for simple responses
            response = self.conversation_chain.invoke({"input": prompt})
            
            # Extract response text
            if isinstance(response, dict) and "response" in response:
                return response["response"]
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error calling simple LLM: {e}")
            return None

    def _get_svg_response(self, context) -> Optional[str]:
        """Generate SVG response if required"""
        try:
            if not self.prompt_chain or not self.svg_chain:
                logger.warning("SVG chains not initialized")
                return None
                
            svg_prompt = self.prompt_chain.invoke({"context": context})
            if hasattr(svg_prompt, 'required') and svg_prompt.required:
                content = self.svg_chain.invoke({"prompt": svg_prompt.prompt})
                return content.svg if hasattr(content, 'svg') else None
            return None
        except Exception as e:
            logger.error(f"Error generating SVG: {e}")
            return None
            
    def _rerank_documents(self, query: str, documents: List[LangchainDocument], top_n: int) -> List[Tuple[LangchainDocument, float]]:
        """Reranks documents using the CrossEncoder and returns top_n with scores."""
        if not self.reranker or not documents:
            logger.warning("Reranker not available or no documents to rerank. Returning original order.")
            return [(doc, 0.0) for doc in documents[:top_n]]

        logger.info(f"Reranking {len(documents)} documents with CrossEncoder...")
        sentence_pairs = [[query, doc.page_content] for doc in documents]
        
        try:
            scores = self.reranker.predict(sentence_pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error during reranker prediction: {e}. Returning original order.")
            return [(doc, 0.0) for doc in documents[:top_n]]

        docs_with_reranker_scores = list(zip(documents, scores))
        docs_with_reranker_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranking complete. Top {top_n} selected.")
        return docs_with_reranker_scores[:top_n]

    def _format_context_for_prompt(self, 
                                   reranked_docs_with_scores: List[Tuple[LangchainDocument, float]],
                                   doc_to_ref_map: Dict[str, int]) -> str:
        """
        Formats reranked documents for the LLM prompt, using numerical references.
        """
        if not reranked_docs_with_scores:
            return "No context snippets were available after reranking."
            
        context_str = ""
        for i, (doc, score) in enumerate(reranked_docs_with_scores):
            metadata = doc.metadata
            source_doc_id = metadata.get('source_doc_id', 'N/A')
            ref_num = doc_to_ref_map.get(source_doc_id, 0) 

            source_ref = (
                f"RefNum: [{ref_num}], "
                f"OrigSourceDocID: {source_doc_id}, "
                f"Paper: {metadata.get('file_name', 'N/A')}, "
                f"Page: {metadata.get('page_number', 'N/A')}, "
                f"Para: {metadata.get('paragraph_number_in_page', 'N/A')}, "
                f"ChunkInPara: {metadata.get('chunk_sequence_in_paragraph', 'N/A')}, "
                f"RerankScore: {score:.4f}" 
            )
            context_str += f"Context Snippet {i+1} ({source_ref}):\n"
            context_str += f"\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
        
        return context_str.strip()

    def get_answer_and_themes(self, 
                            query: str, 
                            collection_name: Optional[str] = None, 
                            n_final_docs_for_llm: int = 10, 
                            initial_mmr_k: int = 100, 
                            initial_mmr_fetch_k: int = 200) -> Dict[str, Any]:
        """
        Main method: retrieves with MMR, reranks with CrossEncoder, then generates answer & themes.
        """
        logger.info(f"Processing query: '{query}'")
        logger.info(f"Config: n_final_docs_for_llm={n_final_docs_for_llm}, initial_mmr_k={initial_mmr_k}, initial_mmr_fetch_k={initial_mmr_fetch_k}")

        # Prepare default empty response structure
        default_empty_response = {
            "answer": "Could not process the query.",
            "themes": [],
            "references": [],
            "retrieved_context_document_ids": [],
            "document_details": [],
            "synthesized_expert_answer": "",
            "svg": None
        }

        # Validate vector store
        if not self.vector_store_service._langchain_chroma_instance:
            logger.error("VectorStoreService not properly initialized.")
            default_empty_response["answer"] = "Error: Knowledge base connection unavailable."
            return default_empty_response
            
        # Retrieve documents using MMR
        mmr_retrieved_docs_tuples = self.vector_store_service.query_with_access_control(
            query_text=query,
            n_results=initial_mmr_k,
            current_collection=collection_name
        )

        if not mmr_retrieved_docs_tuples:
            logger.warning("MMR retrieval found no relevant documents.")
            default_empty_response["answer"] = "Could not find relevant information in the provided documents to answer your query."
            return default_empty_response

        # Rerank documents
        mmr_retrieved_docs = [doc for doc, _ in mmr_retrieved_docs_tuples]
        reranked_docs_with_scores = self._rerank_documents(query, mmr_retrieved_docs, top_n=n_final_docs_for_llm)

        if not reranked_docs_with_scores:
            logger.warning("Reranking yielded no documents.")
            default_empty_response["answer"] = "Could not refine context after initial retrieval."
            return default_empty_response

        logger.info(f"Top {len(reranked_docs_with_scores)} reranked document chunks for LLM context")

        # Create reference mapping
        doc_to_ref_map: Dict[str, int] = {}
        references_list_for_prompt: List[Dict[str, Any]] = []
        current_ref_number = 1
        final_context_doc_ids_for_tracking = []
        source_doc_id_map: Dict[int, str] = {}
        
        for doc, _ in reranked_docs_with_scores:
            source_doc_id = doc.metadata.get('source_doc_id', 'N/A')
            final_context_doc_ids_for_tracking.append(source_doc_id)
            
            if source_doc_id not in doc_to_ref_map:
                doc_to_ref_map[source_doc_id] = current_ref_number
                source_doc_id_map[current_ref_number] = source_doc_id
                references_list_for_prompt.append({
                    "reference_number": current_ref_number,
                    "source_doc_id": source_doc_id,
                    "file_name": doc.metadata.get('file_name', 'N/A')
                })
                current_ref_number += 1

        # Extract document details
        document_details = []
        for doc, score in reranked_docs_with_scores:
            metadata = doc.metadata
            source_doc_id = metadata.get('source_doc_id', 'N/A')
            ref_num = doc_to_ref_map.get(source_doc_id)
            
            if ref_num and ref_num in source_doc_id_map:
                document_details.append({
                    "source_doc_id": source_doc_id,
                    "file_name": metadata.get('file_name', 'N/A'),
                    "extracted_answer": doc.page_content,
                    "page_number": metadata.get('page_number'),
                    "paragraph_number": metadata.get('paragraph_number_in_page'),
                    "rerank_score": score
                })

        # Format context for prompt
        formatted_context = self._format_context_for_prompt(reranked_docs_with_scores, doc_to_ref_map)
        
        logger.info(f"===========================Rag Context=====================================\n{formatted_context}")
        
        # Create main RAG prompt
        prompt_template = f"""You are an intelligent AI Assistant specialized in contextual analysis and information synthesis. You excel at understanding diverse types of queries and adapting your response approach accordingly.

You will receive a user query and relevant context snippets. Each snippet contains identification markers (RefNum, source metadata, relevance scores, etc.) that you must use for proper attribution.

**CORE PRINCIPLES:**
1. **Strict Context Grounding**: Base your response ONLY on the provided context snippets - do not use external knowledge
2. **Query-First Approach**: Always prioritize directly answering what the user asked for
3. **Adaptive Response Style**: Automatically adjust your approach based on the query type:
   - Data extraction queries → Extract and present specific requested information directly
   - Factual queries → Direct, evidence-based answers
   - Analytical queries → Synthesis and interpretation across sources  
   - Comparative queries → Structured comparisons and contrasts
   - Exploratory queries → Comprehensive overviews with multiple perspectives
   - Technical queries → Detailed explanations with precision

**RESPONSE REQUIREMENTS:**

1. **Primary Answer:**
   - FIRST: Directly extract and present exactly what the user asked for
   - Use inline citations with available reference details (e.g., [RefNum, Page: X, Para: Y] or [RefNum] if limited metadata)
   - For data extraction: Present the specific values/information requested in a clear, structured format
   - For analytical queries: Synthesize information across multiple relevant sources
   - Always clearly state if context is insufficient for complete answers

2. **Thematic Analysis** (when multiple sources available):
   - Identify 1-3 key themes or patterns from the context relevant to the query
   - Provide concise summaries for each theme
   - List supporting reference numbers for each theme

3. **Source Transparency:**
   - Maintain clear traceability between all claims and their sources
   - Highlight any limitations or gaps in the available context

**IMPORTANT**: If the user asks for specific data points, values, or information extraction (like "what are the values of A, B, C, D"), focus entirely on extracting and presenting that exact information. Do not provide generic summaries or overarching analysis unless specifically requested.

**User Query:** "{query}"

**Context Snippets:**
{formatted_context}

Please provide a structured response that includes:
- A comprehensive answer with inline citations
- Identified themes with supporting references
- Context assessment
- References used
- Any limitations in the available context"""

        # Call LLM for structured response
        llm_response = self._call_llm_structured(prompt_template)
        logging.info(f"====================LLM STRUCTURED RESPONSE ===============================\n{llm_response}")

        if not llm_response:
            logger.error("LLM call failed or returned no response.")
            default_empty_response["answer"] = "There was an error processing your query with the language model."
            return default_empty_response

        # Process the structured response
        try:
            final_answer = llm_response.answer if llm_response.answer else "No answer was generated by the language model."
            identified_themes = []
            
            # Process themes and add evidence snippets
            for theme in llm_response.identified_themes:
                theme_dict = {
                    "theme_summary": theme.theme_summary,
                    "supporting_reference_numbers": theme.supporting_reference_numbers,
                    "evidence_snippets": []
                }
                
                # Add evidence snippets for each supporting reference
                for ref_num in theme.supporting_reference_numbers:
                    if ref_num in source_doc_id_map:
                        source_doc_id = source_doc_id_map[ref_num]
                        for doc, _ in reranked_docs_with_scores:
                            if doc.metadata.get('source_doc_id') == source_doc_id:
                                theme_dict['evidence_snippets'].append({
                                    "text": doc.page_content,
                                    "page": doc.metadata.get('page_number'),
                                    "paragraph": doc.metadata.get('paragraph_number_in_page'),
                                    "source_doc_id": source_doc_id
                                })
                                break  # Only add one snippet per reference
                
                identified_themes.append(theme_dict)

            # Convert references to dict format
            generated_references = [
                {
                    "reference_number": ref.reference_number,
                    "source_doc_id": ref.source_doc_id,
                    "file_name": ref.file_name
                }
                for ref in llm_response.references
            ]

            # Generate synthesized expert answer using conversation chain
            research_assistant_prompt = f"""You are now acting as a Research Assistant. Given the following user query and the previous LLM answer, synthesize a new expert answer that provides additional insights, clarifications, or a more comprehensive perspective.

User Query: {query}

LLM Answer: {final_answer}

Context: {formatted_context}

Your task is to provide a synthesized expert answer that combines the information from both the user query and the previous answer, offering deeper analysis and expert insights.
NOTE: MAKE SURE THAT ANSWER MUST BE TO THE POINT THE USER QUERY"""

            synthesized_expert_answer = self._call_llm_simple(research_assistant_prompt)
            logging.info(f"====================SYNTHESIZED EXPERT ANSWER ===============================\n{synthesized_expert_answer}")
            if not synthesized_expert_answer:
                synthesized_expert_answer = "Unable to generate synthesized expert answer."

            # Generate SVG if applicable
            svg = self._get_svg_response(synthesized_expert_answer)
            if svg:
                logging.info(f"========================= SVG Generated Successfully =============================")
            else:
                logging.info(f"---------------------------NO SVG Generated --------------------------------------")

            logger.info("RAG processing completed successfully")
            logger.info(f"Generated {len(identified_themes)} themes and {len(generated_references)} references")

            return {
                "answer": final_answer,
                "themes": identified_themes,
                "references": generated_references,
                "retrieved_context_document_ids": final_context_doc_ids_for_tracking,
                "synthesized_expert_answer": synthesized_expert_answer,
                "document_details": document_details,
                "svg": svg
            }

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            default_empty_response["answer"] = f"Error processing response: {str(e)}"
            return default_empty_response


# --- Test Block ---
if __name__ == "__main__":
    logger.info("Testing RAGService with proper chain usage")
    
    vstore_service = VectorStoreService()
    if not vstore_service._langchain_chroma_instance:
        logger.error("CRITICAL: VectorStoreService did not initialize. Aborting.")
        exit()
    
    # Test data
    test_chunks_for_rag = [
        "The novel attention mechanism, 'TransFusion', demonstrates a 15% improvement in NLP task benchmarks by integrating multi-modal inputs effectively.", 
        "Ethical AI frameworks must consider data privacy, algorithmic bias, and societal impact before large-scale deployment of autonomous systems.", 
        "Our proposed 'Contextual Embedding Alignment Protocol' (CEAP) significantly enhances cross-lingual information retrieval from diverse knowledge bases.", 
        "While TransFusion shows promise, its computational overhead for training remains a significant challenge for widespread adoption in resource-constrained environments.", 
        "Bias mitigation techniques in AI, such as adversarial debiasing and data augmentation, are critical for ensuring fairness (Johnson et al., 2023).", 
        "The CEAP method was validated on three distinct language pairs, showing consistent gains over existing SOTA models in zero-shot translation tasks.", 
        "Further research into optimized attention patterns, like those in TransFusion, is essential for next-generation language understanding." 
    ]
    
    test_metadatas_for_rag = [
        {"source_doc_id": "paper_A_2024", "file_name": "transfusion_nlp.pdf", "page_number": 5, "paragraph_number_in_page": 3, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_B_ethics", "file_name": "ethical_ai_frameworks.pdf", "page_number": 12, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_C_ceap", "file_name": "ceap_crosslingual.pdf", "page_number": 7, "paragraph_number_in_page": 4, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_A_2024", "file_name": "transfusion_nlp.pdf", "page_number": 8, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_B_ethics", "file_name": "ethical_ai_frameworks.pdf", "page_number": 15, "paragraph_number_in_page": 2, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_C_ceap", "file_name": "ceap_crosslingual.pdf", "page_number": 9, "paragraph_number_in_page": 2, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_D_attention", "file_name": "future_attention.pdf", "page_number": 2, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
    ]
    
    test_ids_for_rag = [f"rerank_cite_test_v3_{i}" for i in range(len(test_chunks_for_rag))]

    # Clean up and add test documents
    logger.info(f"Cleaning up existing test documents: {test_ids_for_rag}")
    vstore_service.delete_documents(doc_ids=test_ids_for_rag)
    time.sleep(0.5) 
    
    logger.info("Adding test documents for RAG service...")
    vstore_service.add_documents(
        chunks=test_chunks_for_rag, 
        metadatas=test_metadatas_for_rag, 
        doc_ids=test_ids_for_rag
    )
    time.sleep(0.5)
    logger.info(f"Document chunk count after adding test data: {vstore_service.get_collection_count()}")

    # Initialize and test RAG service
    rag_service = RAGService(vector_store_service=vstore_service)

    logger.info("Running RAG Test Query")
    query1 = "What is TransFusion and its significance, including any limitations? Also discuss CEAP."
    
    if rag_service._llm:
        response1 = rag_service.get_answer_and_themes(
            query1, 
            n_final_docs_for_llm=5, 
            initial_mmr_k=10 
        )
        
        if response1:
            logger.info("Test completed successfully")
            print("\nFormatted Response for Query 1:")
            print(json.dumps(response1, indent=2))
        else:
            logger.error("No response generated")
    else:
        logger.error("LLM not properly initialized - skipping test")
    
    logger.info("RAGService Test Complete")