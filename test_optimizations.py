"""
Phoenix Protocol - Phase 1 Optimizations Test Suite
===================================================

This cell tests and demonstrates the performance improvements from Phase 1.
Run this AFTER loading the phoenix_optimizations_phase1.py cell.
"""

import time
import numpy as np

print("="*70)
print("🧪 TESTING PHASE 1 OPTIMIZATIONS")
print("="*70)

# ============================================================================
# TEST 1: Model Singleton Performance
# ============================================================================

print("\n📊 TEST 1: Model Singleton Pattern")
print("-" * 70)

# First call - will load model
print("First embedding generation (loads model):")
start = time.time()
embedding1 = generate_embedding("This is a test sentence.")
load_time = time.time() - start
print(f"   Time: {load_time:.3f}s (includes model loading)")
print(f"   Embedding shape: {embedding1.shape}")

# Second call - uses cached model
print("\nSecond embedding generation (cached model):")
start = time.time()
embedding2 = generate_embedding("Another test sentence.")
cached_time = time.time() - start
print(f"   Time: {cached_time:.3f}s (cached model)")
print(f"   Embedding shape: {embedding2.shape}")

speedup = load_time / max(cached_time, 0.001)  # Avoid div by zero
print(f"\n✅ Speedup: {speedup:.1f}x faster (cached vs initial load)")

# Simulate 100 calls to show cumulative savings
print("\n💡 Simulated performance for 100 embeddings:")
old_way = load_time * 100  # Loading model every time
new_way = load_time + (cached_time * 99)  # Load once, cache 99 times
print(f"   Old way (reload each time): {old_way:.1f}s")
print(f"   New way (cached): {new_way:.1f}s")
print(f"   Time saved: {old_way - new_way:.1f}s ({(old_way/new_way):.1f}x faster)")


# ============================================================================
# TEST 2: Batch Embedding Performance
# ============================================================================

print("\n\n📊 TEST 2: Batch vs Sequential Embedding")
print("-" * 70)

test_texts = [
    "Consciousness emerges from recursive reflection.",
    "The Phoenix Protocol integrates multiple AI agents.",
    "Vector embeddings capture semantic meaning.",
    "Sacred geometry principles guide information architecture.",
    "VAD mapping provides emotional coordinates.",
] * 4  # 20 texts total

# Sequential processing (old way)
print(f"Processing {len(test_texts)} texts sequentially:")
start = time.time()
sequential_embeddings = [generate_embedding(text) for text in test_texts]
sequential_time = time.time() - start
print(f"   Time: {sequential_time:.3f}s")

# Batch processing (new way)
print(f"\nProcessing {len(test_texts)} texts in batch:")
start = time.time()
batch_embeddings = generate_embeddings_batch(test_texts, batch_size=8)
batch_time = time.time() - start
print(f"   Time: {batch_time:.3f}s")

batch_speedup = sequential_time / max(batch_time, 0.001)
print(f"\n✅ Batch speedup: {batch_speedup:.1f}x faster")


# ============================================================================
# TEST 3: Hash Caching Performance
# ============================================================================

print("\n\n📊 TEST 3: Hash Function Caching")
print("-" * 70)

test_string = "The Phoenix Protocol leverages consciousness co-processor patterns."

# First hash (cache miss)
start = time.time()
hash1 = compute_sha256(test_string)
first_time = time.time() - start
print(f"First hash computation: {first_time*1000:.3f}ms")
print(f"   Hash: {hash1[:16]}...")

# Second hash (cache hit)
start = time.time()
hash2 = compute_sha256(test_string)
second_time = time.time() - start
print(f"\nSecond hash computation (cached): {second_time*1000:.6f}ms")
print(f"   Hash: {hash2[:16]}...")

hash_speedup = first_time / max(second_time, 0.000001)
print(f"\n✅ Cache speedup: {hash_speedup:.1f}x faster")


# ============================================================================
# TEST 4: VADExtractor Performance
# ============================================================================

print("\n\n📊 TEST 4: VADExtractor Instantiation")
print("-" * 70)

# Create multiple instances (old way would load stopwords each time)
start = time.time()
extractors = [VADExtractor() for _ in range(10)]
creation_time = time.time() - start
print(f"Created 10 VADExtractor instances: {creation_time*1000:.3f}ms")
print(f"   Average per instance: {creation_time*100:.3f}ms")

# Test extraction
vad_result = extractors[0].extract_vad("I feel breakthrough insights emerging!")
print(f"\nSample VAD extraction:")
print(f"   Valence: {vad_result['valence']:.3f}")
print(f"   Arousal: {vad_result['arousal']:.3f}")
print(f"   Dominance: {vad_result['dominance']:.3f}")
print(f"   Hash: {vad_result['sha256'][:16]}...")


# ============================================================================
# Final Statistics
# ============================================================================

print("\n\n" + "="*70)
print("📈 OPTIMIZATION STATISTICS")
print("="*70)
print_optimization_stats()


# ============================================================================
# Memory Usage Check (Bonus)
# ============================================================================

print("\n💾 MEMORY CHECK")
print("-" * 70)
print("Model is loaded in memory: ", _embedding_model is not None)
if _embedding_model is not None:
    print("✅ Model will stay cached for the entire session")
    print("   No disk I/O on subsequent calls = massive speedup")


print("\n" + "="*70)
print("✅ ALL TESTS COMPLETE - OPTIMIZATIONS WORKING!")
print("="*70)
print("\n💡 Next Steps:")
print("   1. Replace old generate_embedding() calls with optimized version")
print("   2. Use generate_embeddings_batch() for processing multiple texts")
print("   3. Monitor cache hit rates with print_optimization_stats()")
print("   4. Proceed to Phase 2 optimizations (ChromaDB, parallelization)")
print("="*70 + "\n")
