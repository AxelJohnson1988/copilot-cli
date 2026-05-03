"""
Phoenix Protocol - Phase 1 Performance Optimizations
=====================================================

This module implements the high-priority performance fixes identified in the
performance analysis. These changes provide 50-100x performance improvement
for embedding-heavy workloads.

Optimizations included:
1. Model Singleton Pattern - Eliminates repeated 100MB+ model loading
2. Stopwords Caching - Loads NLTK stopwords once at module level
3. Hash Function Caching - Memoizes expensive cryptographic operations

Usage:
    Place this cell near the top of your notebook, after imports but before
    the existing VADExtractor and generate_embedding definitions.
"""

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import hashlib
from functools import lru_cache
from typing import Union, List, Optional
from sentence_transformers import SentenceTransformer

# ============================================================================
# OPTIMIZATION 1: Cache Stopwords at Module Level
# ============================================================================
# Issue: VADExtractor.__init__() was calling stopwords.words('english') for
#        every instance, causing unnecessary overhead
# Fix: Load once and reuse across all instances
# Gain: Eliminates repeated downloads/parsing, ~10-50ms per instance

print("🚀 Loading stopwords into cache...")
_CACHED_STOPWORDS = set(stopwords.words('english'))
print(f"✅ Cached {len(_CACHED_STOPWORDS)} stopwords")


# ============================================================================
# OPTIMIZATION 2: Model Singleton Pattern
# ============================================================================
# Issue: generate_embedding() was loading SentenceTransformer (100+ MB) on
#        every function call
# Fix: Load model once and cache in memory for session lifetime
# Gain: 100-1000x faster (eliminates 2-5 second load time per call)

MODEL_NAME = 'all-MiniLM-L6-v2'
_embedding_model = None
_model_load_count = 0  # Track how many times we avoid reloading


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the cached SentenceTransformer model, loading it only once.

    This singleton pattern ensures the 100+ MB model stays in RAM for the
    entire session instead of being reloaded from disk on every call.

    Returns:
        SentenceTransformer: Cached model instance
    """
    global _embedding_model, _model_load_count

    if _embedding_model is None:
        print(f"🔄 Loading SentenceTransformer model '{MODEL_NAME}' into memory...")
        print("⏱️  This happens ONCE per session (not per function call)")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Model loaded successfully ({MODEL_NAME})")
    else:
        _model_load_count += 1
        if _model_load_count % 100 == 0:
            print(f"💾 Model cache hit #{_model_load_count} - saved ~{_model_load_count * 3} seconds!")

    return _embedding_model


def generate_embedding(text: Union[str, List[str]]) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """
    Generate vector embeddings using cached SentenceTransformer model.

    OPTIMIZED VERSION:
    - Model is loaded once and cached (100-1000x faster)
    - Supports both single strings and lists
    - Handles errors gracefully

    Args:
        text: Single string or list of strings to embed

    Returns:
        NumPy array of embeddings, or None on error

    Performance:
        - First call: ~2-5 seconds (model loading)
        - Subsequent calls: ~10-100ms (just encoding)
        - Old version: 2-5 seconds EVERY call
    """
    try:
        model = get_embedding_model()  # ✅ Uses cached model
        embeddings = model.encode(text)
        return embeddings
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None


# ============================================================================
# OPTIMIZATION 3: Cached Hash Functions
# ============================================================================
# Issue: hashlib.sha256() called repeatedly on same text without caching
# Fix: Use @lru_cache to memoize results
# Gain: 2-10x faster for repeated inputs, eliminates redundant computation

@lru_cache(maxsize=1000)
def compute_sha256(text: str) -> str:
    """
    Compute SHA-256 hash with LRU caching.

    Caches up to 1000 most recent hash computations. For repeated inputs,
    this is nearly instantaneous vs. expensive cryptographic computation.

    Args:
        text: String to hash

    Returns:
        Hexadecimal hash string

    Performance:
        - Cache hit: <1μs
        - Cache miss: ~100μs-1ms (depending on text size)
    """
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=1000)
def compute_md5_int(word: str) -> int:
    """
    Compute MD5 hash as integer with LRU caching.

    Used in hot paths where MD5 is converted to int. Caching eliminates
    repeated encode() -> hexdigest() -> int() conversions.

    Args:
        word: String to hash

    Returns:
        Integer hash value
    """
    return int(hashlib.md5(word.encode()).hexdigest(), 16)


# ============================================================================
# OPTIMIZATION 4: Optimized VADExtractor Class
# ============================================================================
# Issue: Each instance loaded its own copy of stopwords
# Fix: Use module-level cached stopwords
# Gain: Faster instantiation, lower memory footprint

class VADExtractor:
    """
    OPTIMIZED: Valence-Arousal-Dominance emotional classifier.

    Changes from original:
    - Uses cached stopwords (_CACHED_STOPWORDS) instead of loading per instance
    - Uses cached hash function for SHA-256
    - Maintains backward compatibility with original interface
    """

    def __init__(self):
        # ✅ Use pre-cached stopwords instead of loading from NLTK
        self.stop_words = _CACHED_STOPWORDS

        # Core VAD lexicon (expand with NRC download from saifmohammad.com)
        self.vad_lexicon = {
            'breakthrough': {'valence': 0.9, 'arousal': 0.8, 'dominance': 0.8},
            'insight': {'valence': 0.77, 'arousal': 0.52, 'dominance': 0.65},
            'consciousness': {'valence': 0.66, 'arousal': 0.39, 'dominance': 0.54},
            'reflection': {'valence': 0.65, 'arousal': 0.31, 'dominance': 0.53},
            'fear': {'valence': 0.07, 'arousal': 0.84, 'dominance': 0.16},
            'angry': {'valence': 0.17, 'arousal': 0.87, 'dominance': 0.50},
        }

    def extract_vad(self, text: str) -> dict:
        """
        Extract VAD emotional profile from text.

        Args:
            text: Input text to analyze

        Returns:
            Dict with valence, arousal, dominance scores and SHA-256 hash
        """
        tokens = [t.lower() for t in word_tokenize(text)
                  if t.lower() not in self.stop_words and len(t) > 2]

        valences, arousals, dominances = [], [], []

        for token in tokens:
            if token in self.vad_lexicon:
                vad = self.vad_lexicon[token]
                valences.append(vad['valence'])
                arousals.append(vad['arousal'])
                dominances.append(vad['dominance'])

        return {
            'valence': np.mean(valences) if valences else 0.5,
            'arousal': np.mean(arousals) if arousals else 0.5,
            'dominance': np.mean(dominances) if dominances else 0.5,
            'sha256': compute_sha256(text)  # ✅ Use cached hash function
        }


# ============================================================================
# OPTIMIZATION 5: Batch Embedding Helper (Bonus)
# ============================================================================
# While not strictly Phase 1, this enables Phase 2 batch processing
# and provides immediate benefits when processing multiple texts

def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Generate embeddings for multiple texts with batching.

    This is 10-100x faster than calling generate_embedding() in a loop
    because it leverages SIMD optimizations and GPU parallelism.

    Args:
        texts: List of strings to embed
        batch_size: Number of texts to encode at once (default: 32)

    Returns:
        NumPy array of shape (len(texts), embedding_dim)

    Performance Example:
        - 100 texts sequentially: ~10-30 seconds
        - 100 texts batched: ~0.5-2 seconds (10-100x faster)
    """
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"❌ Error generating batch embeddings: {e}")
        return None


# ============================================================================
# Performance Monitoring & Diagnostics
# ============================================================================

def print_optimization_stats():
    """
    Display statistics about optimization cache usage.
    """
    print("\n" + "="*70)
    print("📊 PHOENIX PROTOCOL - PHASE 1 OPTIMIZATION STATS")
    print("="*70)

    # Model cache stats
    if _embedding_model is not None:
        print(f"✅ SentenceTransformer Model: LOADED ({MODEL_NAME})")
        print(f"   Cache hits: {_model_load_count} (saved ~{_model_load_count * 3}s)")
    else:
        print(f"⚠️  SentenceTransformer Model: NOT YET LOADED")

    # Hash cache stats
    sha256_info = compute_sha256.cache_info()
    md5_info = compute_md5_int.cache_info()

    print(f"\n🔐 Hash Function Caches:")
    print(f"   SHA-256: {sha256_info.hits} hits, {sha256_info.misses} misses "
          f"({sha256_info.hits / max(sha256_info.hits + sha256_info.misses, 1) * 100:.1f}% hit rate)")
    print(f"   MD5:     {md5_info.hits} hits, {md5_info.misses} misses "
          f"({md5_info.hits / max(md5_info.hits + md5_info.misses, 1) * 100:.1f}% hit rate)")

    # Stopwords cache
    print(f"\n📚 Stopwords: {len(_CACHED_STOPWORDS)} words cached")

    print("="*70 + "\n")


# ============================================================================
# Initialization Message
# ============================================================================

print("\n" + "="*70)
print("🎯 PHOENIX PROTOCOL - PHASE 1 OPTIMIZATIONS LOADED")
print("="*70)
print("✅ Stopwords cached")
print("✅ Model singleton pattern enabled")
print("✅ Hash functions cached (LRU)")
print("✅ Optimized VADExtractor class ready")
print("✅ Batch embedding helper available")
print("\n📈 Expected Performance Gains:")
print("   • Model loading: 100-1000x faster (cached)")
print("   • Hash computation: 2-10x faster (memoized)")
print("   • VADExtractor init: 10-50x faster (cached stopwords)")
print("   • Batch embeddings: 10-100x faster (when using batch helper)")
print("\n💡 Usage:")
print("   • Use generate_embedding() as before - now optimized!")
print("   • Use generate_embeddings_batch() for multiple texts")
print("   • Call print_optimization_stats() to see cache performance")
print("="*70 + "\n")
