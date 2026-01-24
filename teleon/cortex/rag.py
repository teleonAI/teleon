"""
RAG (Retrieval-Augmented Generation) - Document storage and retrieval for LLM context.

Provides utilities for:
- Document chunking (fixed, semantic, recursive)
- Chunk storage and retrieval
- Hybrid search (semantic + keyword)
- Chunk reranking
- Context fusion
"""

from typing import List, Dict, Optional, Any, Tuple
import hashlib
import re
import logging

from teleon.cortex.token_manager import TokenCounter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Intelligent document chunking for RAG.
    
    Strategies:
    - fixed: Simple fixed-size chunks with overlap
    - semantic: Respect paragraph and section boundaries
    - recursive: Hierarchical chunking for long documents
    
    Example:
        ```python
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        
        # Fixed chunking
        chunks = chunker.chunk_document(
            document=long_text,
            strategy="fixed"
        )
        
        # Semantic chunking
        chunks = chunker.chunk_by_semantic_units(
            document=long_text,
            target_size=500
        )
        ```
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            token_counter: Token counter (creates default if None)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = token_counter or TokenCounter()
    
    def chunk_document(
        self,
        document: str,
        strategy: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document using specified strategy.
        
        Args:
            document: Document text
            strategy: Chunking strategy
            metadata: Metadata to attach to all chunks
        
        Returns:
            List of chunk dicts with 'text', 'index', 'metadata'
        """
        if not document or not document.strip():
            return []
        
        if strategy == "fixed":
            chunks = self._fixed_chunking(document)
        elif strategy == "semantic":
            chunks = self._semantic_chunking(document)
        elif strategy == "recursive":
            chunks = self._recursive_chunking(document)
        else:
            logger.warning(f"Unknown strategy {strategy}, using semantic")
            chunks = self._semantic_chunking(document)
        
        # Add metadata and index
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                "text": chunk_text,
                "index": i,
                "total_chunks": len(chunks),
                "metadata": metadata or {}
            }
            result.append(chunk_dict)
        
        return result
    
    def _fixed_chunking(self, document: str) -> List[str]:
        """Simple fixed-size chunking with overlap."""
        return self.token_counter.split_by_tokens(
            document,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
    
    def _semantic_chunking(self, document: str) -> List[str]:
        """
        Chunk by semantic units (paragraphs, sections).
        
        Respects:
        - Paragraph breaks (\n\n)
        - Section headers (# headings)
        - Sentence boundaries
        """
        # Split by paragraphs first
        paragraphs = document.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self.token_counter.count_tokens(para)
            
            # If paragraph alone exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = self._split_sentences(para)
                sent_chunk = []
                sent_tokens = 0
                
                for sent in sentences:
                    sent_tok = self.token_counter.count_tokens(sent)
                    
                    if sent_tokens + sent_tok > self.chunk_size:
                        if sent_chunk:
                            chunks.append(" ".join(sent_chunk))
                        sent_chunk = [sent]
                        sent_tokens = sent_tok
                    else:
                        sent_chunk.append(sent)
                        sent_tokens += sent_tok
                
                if sent_chunk:
                    chunks.append(" ".join(sent_chunk))
            
            # If adding paragraph fits, add it
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_tokens += para_tokens
            
            # Otherwise, save current and start new
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
        
        # Add remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _recursive_chunking(self, document: str, level: int = 0) -> List[str]:
        """
        Recursive hierarchical chunking.
        
        Strategy:
        1. Try to split by sections
        2. If sections too large, split by paragraphs
        3. If paragraphs too large, split by sentences
        4. If sentences too large, use fixed chunking
        """
        tokens = self.token_counter.count_tokens(document)
        
        # Base case: fits in one chunk
        if tokens <= self.chunk_size:
            return [document]
        
        # Try splitting by sections (marked with ## or ###)
        if level == 0:
            sections = re.split(r'\n##+ ', document)
            if len(sections) > 1:
                chunks = []
                for section in sections:
                    chunks.extend(self._recursive_chunking(section, level + 1))
                return chunks
        
        # Try splitting by paragraphs
        if level <= 1:
            paragraphs = document.split("\n\n")
            if len(paragraphs) > 1:
                chunks = []
                current = []
                current_tokens = 0
                
                for para in paragraphs:
                    para_tokens = self.token_counter.count_tokens(para)
                    
                    if current_tokens + para_tokens > self.chunk_size:
                        if current:
                            chunks.extend(
                                self._recursive_chunking("\n\n".join(current), level + 1)
                            )
                        current = [para]
                        current_tokens = para_tokens
                    else:
                        current.append(para)
                        current_tokens += para_tokens
                
                if current:
                    chunks.extend(
                        self._recursive_chunking("\n\n".join(current), level + 1)
                    )
                
                return chunks
        
        # Fall back to fixed chunking
        return self._fixed_chunking(document)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class ChunkRanker:
    """
    Rerank retrieved chunks for relevance.
    
    Strategies:
    - similarity: Use embedding similarity scores
    - hybrid: Combine semantic and keyword matching
    - diversity: Ensure diverse chunks
    - position: Consider chunk position in document
    
    Example:
        ```python
        ranker = ChunkRanker()
        
        # Rerank chunks
        reranked = ranker.rerank_chunks(
            query="What is machine learning?",
            chunks=chunks,
            scores=similarity_scores,
            method="hybrid"
        )
        ```
    """
    
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize ranker.
        
        Args:
            token_counter: Token counter instance
        """
        self.token_counter = token_counter or TokenCounter()
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        scores: List[float],
        method: str = "hybrid",
        top_k: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank chunks by relevance.
        
        Args:
            query: Search query
            chunks: List of chunk dicts
            scores: Initial similarity scores
            method: Reranking method
            top_k: Return top K chunks
        
        Returns:
            List of (chunk, final_score) tuples
        """
        if not chunks:
            return []
        
        if method == "similarity":
            reranked = list(zip(chunks, scores))
        elif method == "hybrid":
            reranked = self._hybrid_rerank(query, chunks, scores)
        elif method == "diversity":
            reranked = self._diversity_rerank(chunks, scores)
        else:
            logger.warning(f"Unknown method {method}, using similarity")
            reranked = list(zip(chunks, scores))
        
        # Sort by score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _hybrid_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        similarity_scores: List[float]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid reranking: semantic + keyword + position.
        
        Combines:
        - Semantic similarity (60%)
        - Keyword overlap (30%)
        - Position bonus (10%)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        reranked = []
        
        for chunk, sim_score in zip(chunks, similarity_scores):
            text = chunk["text"].lower()
            text_words = set(text.split())
            
            # Keyword overlap score
            overlap = len(query_words & text_words)
            keyword_score = min(1.0, overlap / max(len(query_words), 1))
            
            # Position bonus (earlier chunks get slight boost)
            index = chunk.get("index", 0)
            total = chunk.get("total_chunks", 1)
            position_score = 1.0 - (index / max(total, 1)) * 0.3
            
            # Combined score
            final_score = (
                sim_score * 0.6 +
                keyword_score * 0.3 +
                position_score * 0.1
            )
            
            reranked.append((chunk, final_score))
        
        return reranked
    
    def _diversity_rerank(
        self,
        chunks: List[Dict[str, Any]],
        scores: List[float]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank ensuring diversity (avoid redundant chunks).
        
        Strategy: Penalize chunks similar to already-selected chunks.
        """
        if not chunks:
            return []
        
        # Start with highest scoring chunk
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [chunk_scores[0]]
        remaining = chunk_scores[1:]
        
        # Iteratively select diverse chunks
        while remaining:
            best_score = -1
            best_idx = 0
            
            for i, (chunk, score) in enumerate(remaining):
                # Calculate diversity penalty
                penalty = self._calculate_diversity_penalty(
                    chunk, [c for c, _ in selected]
                )
                
                adjusted_score = score * (1 - penalty * 0.5)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = i
            
            # Add best diverse chunk
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected
    
    def _calculate_diversity_penalty(
        self,
        chunk: Dict[str, Any],
        selected_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate diversity penalty (0-1).
        
        Higher penalty = more similar to already-selected chunks.
        """
        if not selected_chunks:
            return 0.0
        
        chunk_text = chunk["text"].lower()
        chunk_words = set(chunk_text.split())
        
        max_overlap = 0.0
        
        for selected in selected_chunks:
            selected_text = selected["text"].lower()
            selected_words = set(selected_text.split())
            
            # Calculate word overlap
            overlap = len(chunk_words & selected_words)
            overlap_ratio = overlap / max(len(chunk_words), 1)
            
            max_overlap = max(max_overlap, overlap_ratio)
        
        return max_overlap


class ContextFusion:
    """
    Fuse multiple retrieved chunks into coherent context.
    
    Methods:
    - concat: Simple concatenation
    - summarize: Summarize chunks
    - synthesize: Synthesize into coherent text
    - deduplicate: Remove redundant information
    
    Example:
        ```python
        fusion = ContextFusion(token_counter)
        
        # Fuse chunks
        context = fusion.fuse_chunks(
            chunks=chunks,
            max_tokens=1000,
            method="deduplicate"
        )
        ```
    """
    
    def __init__(self, token_counter: TokenCounter):
        """
        Initialize context fusion.
        
        Args:
            token_counter: Token counter instance
        """
        self.token_counter = token_counter
    
    def fuse_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int,
        method: str = "concat",
        query: Optional[str] = None
    ) -> str:
        """
        Fuse chunks into coherent context.
        
        Args:
            chunks: List of chunk dicts
            max_tokens: Maximum tokens for result
            method: Fusion method
            query: Optional query for context
        
        Returns:
            Fused context string
        """
        if not chunks:
            return ""
        
        if method == "concat":
            return self._concatenate_chunks(chunks, max_tokens)
        elif method == "deduplicate":
            return self._deduplicate_and_concat(chunks, max_tokens)
        elif method == "summarize":
            return self._summarize_chunks(chunks, max_tokens)
        else:
            logger.warning(f"Unknown method {method}, using concat")
            return self._concatenate_chunks(chunks, max_tokens)
    
    def _concatenate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> str:
        """Simple concatenation with token limit."""
        parts = []
        total_tokens = 0
        
        for chunk in chunks:
            text = chunk["text"]
            tokens = self.token_counter.count_tokens(text)
            
            if total_tokens + tokens <= max_tokens:
                parts.append(text)
                total_tokens += tokens
            else:
                # Try to fit truncated version
                remaining = max_tokens - total_tokens
                if remaining > 50:
                    truncated = self.token_counter.truncate_to_tokens(
                        text, remaining
                    )
                    parts.append(truncated)
                break
        
        return "\n\n".join(parts)
    
    def _deduplicate_and_concat(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> str:
        """Deduplicate chunks before concatenation."""
        # Simple deduplication: remove chunks with high similarity
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk["text"]
            
            # Create fingerprint (first 100 chars)
            fingerprint = text[:100].lower().strip()
            
            if fingerprint not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(fingerprint)
        
        return self._concatenate_chunks(unique_chunks, max_tokens)
    
    def _summarize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> str:
        """
        Summarize chunks into compact context.
        
        Simple extractive summarization: extracts key sentences from chunks.
        This is a basic implementation that works but could be enhanced with
        LLM-based abstractive summarization for better quality.
        """
        # Extract key sentences from each chunk
        extracted = []
        
        for chunk in chunks:
            text = chunk["text"]
            
            # Get first sentence (often contains key info)
            sentences = text.split(". ")
            if sentences:
                extracted.append(sentences[0] + ".")
        
        # Concatenate extractions
        result = " ".join(extracted)
        
        # Truncate if needed
        if self.token_counter.count_tokens(result) > max_tokens:
            result = self.token_counter.truncate_to_tokens(result, max_tokens)
        
        return result


class RAGMemory:
    """
    Main RAG interface for document storage and retrieval.
    
    Combines:
    - Document chunking
    - Chunk storage in semantic memory
    - Intelligent retrieval
    - Reranking and fusion
    
    Example:
        ```python
        from teleon.cortex import create_cortex
        from teleon.cortex.rag import RAGMemory
        
        cortex = await create_cortex(agent_id="agent-123")
        rag = RAGMemory(cortex)
        
        # Store document
        chunk_ids = await rag.store_document(
            document="Long article about AI...",
            metadata={'source': 'article.pdf', 'topic': 'AI'}
        )
        
        # Retrieve for query
        context = await rag.retrieve_for_query(
            query="What is machine learning?",
            num_chunks=3
        )
        ```
    """
    
    def __init__(self, cortex, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize RAG memory.
        
        Args:
            cortex: CortexMemory instance
            chunk_size: Chunk size in tokens
            overlap: Overlap between chunks
        """
        self.cortex = cortex
        self.token_counter = TokenCounter()
        self.chunker = DocumentChunker(chunk_size, overlap, self.token_counter)
        self.ranker = ChunkRanker(self.token_counter)
        self.fusion = ContextFusion(self.token_counter)
    
    async def store_document(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: str = "semantic"
    ) -> List[str]:
        """
        Chunk and store document in semantic memory.
        
        Args:
            document: Document text
            metadata: Metadata for all chunks
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            chunking_strategy: Chunking strategy
        
        Returns:
            List of chunk IDs
        """
        # Use custom chunk size if provided
        if chunk_size or chunk_overlap:
            chunker = DocumentChunker(
                chunk_size or self.chunker.chunk_size,
                chunk_overlap or self.chunker.overlap,
                self.token_counter
            )
        else:
            chunker = self.chunker
        
        # Chunk document
        chunks = chunker.chunk_document(
            document,
            strategy=chunking_strategy,
            metadata=metadata
        )
        
        logger.info(f"Chunked document into {len(chunks)} chunks")
        
        # Store each chunk in semantic memory
        chunk_ids = []
        
        for chunk in chunks:
            # Create chunk metadata
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": chunk["index"],
                "total_chunks": chunk["total_chunks"],
                "is_chunk": True,
            }
            
            # Store in semantic memory
            entry_id = await self.cortex.semantic.store(
                content=chunk["text"],
                metadata=chunk_metadata,
                tags=metadata.get("tags", []) if metadata else []
            )
            
            chunk_ids.append(entry_id)
        
        logger.info(f"Stored {len(chunk_ids)} chunks in semantic memory")
        
        return chunk_ids
    
    async def retrieve_for_query(
        self,
        query: str,
        num_chunks: int = 5,
        rerank: bool = True,
        fusion_method: str = "deduplicate",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Retrieve and prepare context for LLM query.
        
        Args:
            query: Search query
            num_chunks: Number of chunks to retrieve
            rerank: Whether to rerank results
            fusion_method: Method for fusing chunks
            max_tokens: Maximum tokens for result
        
        Returns:
            Fused context string
        """
        # Search semantic memory (use 0.0 threshold to get all results, we'll filter later)
        results = await self.cortex.semantic.search(
            query,
            limit=num_chunks * 2 if rerank else num_chunks,
            min_similarity=0.0
        )
        
        if not results:
            return ""
        
        # Extract chunks and scores
        chunks = []
        scores = []
        
        for entry, score in results:
            chunk_dict = {
                "text": entry.content,
                "metadata": entry.metadata,
                "index": entry.metadata.get("chunk_index", 0),
                "total_chunks": entry.metadata.get("total_chunks", 1),
            }
            chunks.append(chunk_dict)
            scores.append(score)
        
        # Rerank if requested
        if rerank:
            reranked = self.ranker.rerank_chunks(
                query, chunks, scores, method="hybrid", top_k=num_chunks
            )
            chunks = [chunk for chunk, _ in reranked]
        else:
            chunks = chunks[:num_chunks]
        
        # Fuse chunks
        if max_tokens is None:
            max_tokens = self.chunker.chunk_size * num_chunks
        
        context = self.fusion.fuse_chunks(
            chunks,
            max_tokens=max_tokens,
            method=fusion_method,
            query=query
        )
        
        return context
    
    async def hybrid_search(
        self,
        query: str,
        num_chunks: int = 5,
        semantic_weight: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search: semantic + keyword.
        
        Args:
            query: Search query
            num_chunks: Number of chunks to return
            semantic_weight: Weight for semantic score (0-1)
        
        Returns:
            List of (chunk_text, combined_score) tuples
        """
        keyword_weight = 1.0 - semantic_weight
        
        # Semantic search
        semantic_results = await self.cortex.semantic.search(
            query, limit=num_chunks * 2, min_similarity=0.0
        )
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        # Combine scores
        combined = []
        
        for entry, sem_score in semantic_results:
            text = entry.content.lower()
            text_words = set(text.split())
            
            # Keyword score
            overlap = len(query_words & text_words)
            keyword_score = min(1.0, overlap / max(len(query_words), 1))
            
            # Combined score
            final_score = (
                sem_score * semantic_weight +
                keyword_score * keyword_weight
            )
            
            combined.append((entry.content, final_score))
        
        # Sort and return top K
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:num_chunks]


__all__ = [
    "RAGMemory",
    "DocumentChunker",
    "ChunkRanker",
    "ContextFusion",
]

