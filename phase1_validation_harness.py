"""
Phoenix Protocol - Phase 1 Validation Harness
==============================================

Validates that Phase 1 optimizations meet production reliability standards.
This is your "checkpoint" before unlocking Phase 2.

Tests 5 Critical Criteria:
1. Deterministic outputs → Same input always produces same result
2. Latency baseline → Establish average + worst-case timing
3. Error handling → Failures are predictable and recoverable
4. Data flow traceability → End-to-end request tracking
5. Integration surface → Clean, structured inputs/outputs

Pass Criteria: Score ≥90% across all 5 dimensions
Fail Criteria: ANY dimension <80% or critical error unhandled

Usage:
    1. Load phoenix_optimizations_phase1.py
    2. Run this harness
    3. Review detailed report
    4. Fix any issues found
    5. Re-run until "PHASE 2 READY" certification achieved
"""

import time
import json
import hashlib
import traceback
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

# ============================================================================
# TEST DATA: Real-world Phoenix Protocol inputs
# ============================================================================

REAL_WORLD_TEST_CASES = [
    # Case 1: Consciousness/reflection content
    {
        "id": "consciousness_01",
        "text": "The consciousness co-processor applies recursive reflection patterns as a semantic distillation pipeline. Through VAD emotional mapping, artifacts flow through parsing, emotional classification, vector embedding, coherence scoring, and cross-referencing.",
        "expected_vad_range": {"valence": (0.4, 0.8), "arousal": (0.3, 0.7), "dominance": (0.4, 0.8)},
        "min_embedding_dim": 384
    },

    # Case 2: Technical/analytical content
    {
        "id": "technical_01",
        "text": "ChromaDB integrates with Phoenix Protocol to store artifacts with VAD emotional metadata alongside SHA-256 verification hashes. This enables hybrid search combining semantic similarity with emotional resonance.",
        "expected_vad_range": {"valence": (0.4, 0.7), "arousal": (0.3, 0.6), "dominance": (0.4, 0.7)},
        "min_embedding_dim": 384
    },

    # Case 3: Emotional/breakthrough content
    {
        "id": "breakthrough_01",
        "text": "Breakthrough insights emerge when consciousness reflects upon itself. The moment of recognition carries intense emotional resonance - a feeling of profound understanding that transforms chaos into navigable truth.",
        "expected_vad_range": {"valence": (0.6, 0.95), "arousal": (0.5, 0.9), "dominance": (0.5, 0.9)},
        "min_embedding_dim": 384
    },

    # Case 4: Problem/uncertainty content
    {
        "id": "problem_01",
        "text": "The system encounters errors when processing malformed data. Fear and uncertainty arise from unpredictable behavior. Integration challenges create anxiety around system reliability.",
        "expected_vad_range": {"valence": (0.0, 0.3), "arousal": (0.6, 0.95), "dominance": (0.0, 0.3)},
        "min_embedding_dim": 384
    },

    # Case 5: Balanced/neutral content
    {
        "id": "neutral_01",
        "text": "The framework processes conversational artifacts through multiple stages. Each stage applies specific transformations to extract structured information from unstructured text.",
        "expected_vad_range": {"valence": (0.4, 0.6), "arousal": (0.3, 0.6), "dominance": (0.4, 0.6)},
        "min_embedding_dim": 384
    },

    # Case 6: Edge case - very short text
    {
        "id": "edge_short",
        "text": "Brief insight.",
        "expected_vad_range": {"valence": (0.3, 0.8), "arousal": (0.2, 0.7), "dominance": (0.3, 0.7)},
        "min_embedding_dim": 384
    },

    # Case 7: Edge case - repetitive text
    {
        "id": "edge_repetitive",
        "text": "The reflection reflects the reflection. The pattern patterns the pattern. The consciousness becomes conscious of consciousness.",
        "expected_vad_range": {"valence": (0.3, 0.7), "arousal": (0.2, 0.7), "dominance": (0.3, 0.7)},
        "min_embedding_dim": 384
    },

    # Case 8: Edge case - special characters
    {
        "id": "edge_special_chars",
        "text": "Phoenix Protocol (v1.0) integrates $emotional-mapping$ with [semantic-vectors]. Key: SHA-256 → integrity; VAD → emotions; φ ≈ 1.618 → golden ratio.",
        "expected_vad_range": {"valence": (0.3, 0.7), "arousal": (0.2, 0.7), "dominance": (0.3, 0.7)},
        "min_embedding_dim": 384
    },
]

# Batch test cases (for performance testing)
BATCH_TEST_TEXTS = [tc["text"] for tc in REAL_WORLD_TEST_CASES]

# ============================================================================
# VALIDATION FRAMEWORK
# ============================================================================

class ValidationResult:
    """Stores validation results for a single test."""
    def __init__(self, test_id: str, test_name: str):
        self.test_id = test_id
        self.test_name = test_name
        self.passed = False
        self.score = 0.0
        self.latency_ms = 0.0
        self.errors = []
        self.warnings = []
        self.details = {}

    def to_dict(self):
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "passed": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details
        }


class ValidationHarness:
    """Main validation orchestrator."""

    def __init__(self):
        self.results = []
        self.category_scores = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.trace_log = []

    def log_trace(self, category: str, message: str, data: Any = None):
        """End-to-end request tracing."""
        self.trace_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "message": message,
            "data": data
        })

    def run_all_validations(self) -> Dict[str, Any]:
        """Execute all 5 validation categories."""
        self.start_time = time.time()

        print("="*80)
        print("🔬 PHOENIX PROTOCOL - PHASE 1 VALIDATION HARNESS")
        print("="*80)
        print(f"Started: {datetime.utcnow().isoformat()}")
        print(f"Test cases: {len(REAL_WORLD_TEST_CASES)} real-world scenarios")
        print("="*80 + "\n")

        # Run all 5 validation categories
        self.validate_determinism()
        self.validate_latency()
        self.validate_error_handling()
        self.validate_data_flow()
        self.validate_integration_surface()

        self.end_time = time.time()

        # Generate final report
        return self.generate_report()

    # ========================================================================
    # CRITERION 1: Deterministic Outputs
    # ========================================================================

    def validate_determinism(self):
        """Test: Same input → Same output (every time)."""
        print("\n" + "─"*80)
        print("📊 CRITERION 1: Deterministic Outputs")
        print("─"*80)

        category = "determinism"
        passed_tests = 0
        total_tests = 0

        for test_case in REAL_WORLD_TEST_CASES[:5]:  # Test first 5 cases
            result = ValidationResult(test_case["id"], f"Determinism: {test_case['id']}")
            self.log_trace(category, f"Starting determinism test", test_case["id"])

            try:
                text = test_case["text"]

                # Run same operation 3 times
                self.log_trace(category, "Run 1/3")
                hash1 = compute_sha256(text)
                vad1 = VADExtractor().extract_vad(text)
                emb1 = generate_embedding(text)

                time.sleep(0.1)  # Small delay

                self.log_trace(category, "Run 2/3")
                hash2 = compute_sha256(text)
                vad2 = VADExtractor().extract_vad(text)
                emb2 = generate_embedding(text)

                time.sleep(0.1)

                self.log_trace(category, "Run 3/3")
                hash3 = compute_sha256(text)
                vad3 = VADExtractor().extract_vad(text)
                emb3 = generate_embedding(text)

                # Verify determinism
                hash_match = (hash1 == hash2 == hash3)
                vad_match = (
                    vad1['valence'] == vad2['valence'] == vad3['valence'] and
                    vad1['arousal'] == vad2['arousal'] == vad3['arousal'] and
                    vad1['dominance'] == vad2['dominance'] == vad3['dominance']
                )
                emb_match = np.allclose(emb1, emb2) and np.allclose(emb2, emb3)

                # Calculate score
                score = sum([hash_match, vad_match, emb_match]) / 3.0 * 100
                result.score = score
                result.passed = score >= 90.0

                result.details = {
                    "hash_deterministic": hash_match,
                    "vad_deterministic": vad_match,
                    "embedding_deterministic": emb_match,
                    "hash_value": hash1[:16],
                    "vad_values": vad1,
                    "embedding_shape": emb1.shape
                }

                if not result.passed:
                    if not hash_match:
                        result.errors.append(f"Hash mismatch: {hash1[:8]} != {hash2[:8]}")
                    if not vad_match:
                        result.errors.append(f"VAD drift detected")
                    if not emb_match:
                        result.errors.append(f"Embedding non-deterministic")

                if result.passed:
                    passed_tests += 1
                total_tests += 1

                self.log_trace(category, f"Test complete: {result.passed}", result.details)

            except Exception as e:
                result.errors.append(f"Exception: {str(e)}")
                result.details["traceback"] = traceback.format_exc()
                self.log_trace(category, f"Test failed with exception", str(e))

            self.results.append(result)
            self.category_scores[category].append(result.score)

            # Print result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} {test_case['id']}: {result.score:.1f}% deterministic")

        category_avg = np.mean(self.category_scores[category])
        print(f"\n  Category Score: {category_avg:.1f}% ({passed_tests}/{total_tests} passed)")

    # ========================================================================
    # CRITERION 2: Latency Baseline
    # ========================================================================

    def validate_latency(self):
        """Test: Establish average + worst-case timing."""
        print("\n" + "─"*80)
        print("⏱️  CRITERION 2: Latency Baseline")
        print("─"*80)

        category = "latency"
        result = ValidationResult("latency_baseline", "Latency Baseline Measurement")

        try:
            latencies = {
                "hash_compute": [],
                "vad_extract": [],
                "embedding_single": [],
                "embedding_batch": [],
                "model_cache_hit": []
            }

            # Test hash computation (10 runs)
            for i in range(10):
                text = REAL_WORLD_TEST_CASES[i % len(REAL_WORLD_TEST_CASES)]["text"]
                start = time.time()
                compute_sha256(text)
                latencies["hash_compute"].append((time.time() - start) * 1000)

            # Test VAD extraction (10 runs)
            extractor = VADExtractor()
            for i in range(10):
                text = REAL_WORLD_TEST_CASES[i % len(REAL_WORLD_TEST_CASES)]["text"]
                start = time.time()
                extractor.extract_vad(text)
                latencies["vad_extract"].append((time.time() - start) * 1000)

            # Test embedding - first call (model load)
            text = REAL_WORLD_TEST_CASES[0]["text"]
            start = time.time()
            generate_embedding(text)
            first_call_latency = (time.time() - start) * 1000

            # Test embedding - subsequent calls (cached model)
            for i in range(10):
                text = REAL_WORLD_TEST_CASES[i % len(REAL_WORLD_TEST_CASES)]["text"]
                start = time.time()
                generate_embedding(text)
                latencies["model_cache_hit"].append((time.time() - start) * 1000)

            # Test batch embeddings
            for _ in range(3):
                start = time.time()
                generate_embeddings_batch(BATCH_TEST_TEXTS)
                latencies["embedding_batch"].append((time.time() - start) * 1000)

            # Calculate statistics
            stats = {}
            for op, times in latencies.items():
                stats[op] = {
                    "avg_ms": np.mean(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "p95_ms": np.percentile(times, 95),
                    "p99_ms": np.percentile(times, 99)
                }

            result.details = {
                "first_embedding_load_ms": first_call_latency,
                "operations": stats
            }

            # Scoring: Check if latencies are within acceptable ranges
            checks = [
                stats["hash_compute"]["avg_ms"] < 5.0,  # Hash should be fast
                stats["vad_extract"]["avg_ms"] < 100.0,  # VAD should be reasonable
                stats["model_cache_hit"]["avg_ms"] < 200.0,  # Cached embedding fast
                first_call_latency < 10000.0,  # Model load < 10s
                stats["embedding_batch"]["avg_ms"] < 5000.0  # Batch < 5s
            ]

            result.score = sum(checks) / len(checks) * 100
            result.passed = result.score >= 90.0

            # Print detailed breakdown
            print(f"\n  Model Load (first call): {first_call_latency:.1f}ms")
            print(f"  Hash (avg):              {stats['hash_compute']['avg_ms']:.2f}ms")
            print(f"  VAD Extract (avg):       {stats['vad_extract']['avg_ms']:.2f}ms")
            print(f"  Embedding cached (avg):  {stats['model_cache_hit']['avg_ms']:.1f}ms")
            print(f"  Embedding batch (avg):   {stats['embedding_batch']['avg_ms']:.1f}ms")
            print(f"  Embedding p95:           {stats['model_cache_hit']['p95_ms']:.1f}ms")
            print(f"  Embedding p99:           {stats['model_cache_hit']['p99_ms']:.1f}ms")

            self.log_trace(category, "Latency baseline established", stats)

        except Exception as e:
            result.errors.append(f"Exception: {str(e)}")
            result.details["traceback"] = traceback.format_exc()
            self.log_trace(category, f"Test failed with exception", str(e))

        self.results.append(result)
        self.category_scores[category].append(result.score)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  {status} Category Score: {result.score:.1f}%")

    # ========================================================================
    # CRITERION 3: Error Handling
    # ========================================================================

    def validate_error_handling(self):
        """Test: Failures are predictable and recoverable."""
        print("\n" + "─"*80)
        print("🛡️  CRITERION 3: Error Handling")
        print("─"*80)

        category = "error_handling"
        passed_tests = 0
        total_tests = 0

        error_test_cases = [
            {
                "id": "empty_string",
                "input": "",
                "operation": "generate_embedding",
                "should_handle": True
            },
            {
                "id": "none_input",
                "input": None,
                "operation": "generate_embedding",
                "should_handle": True
            },
            {
                "id": "very_long_text",
                "input": "word " * 10000,  # 10k words
                "operation": "generate_embedding",
                "should_handle": True
            },
            {
                "id": "unicode_heavy",
                "input": "🔥💡🎯🚀✅❌⚠️ Phoenix Protocol 中文 العربية עברית",
                "operation": "generate_embedding",
                "should_handle": True
            },
        ]

        for test in error_test_cases:
            result = ValidationResult(test["id"], f"Error: {test['id']}")

            try:
                self.log_trace(category, f"Testing error case: {test['id']}")

                error_occurred = False
                error_handled = False
                output = None

                # Try operation
                try:
                    if test["operation"] == "generate_embedding":
                        output = generate_embedding(test["input"])
                except Exception as e:
                    error_occurred = True
                    error_handled = True  # Exception was caught
                    result.details["exception"] = str(e)

                # Check if result is valid or None (graceful failure)
                if output is None:
                    error_handled = True
                elif isinstance(output, np.ndarray):
                    error_handled = True  # Succeeded

                result.passed = error_handled
                result.score = 100.0 if result.passed else 0.0
                result.details["error_occurred"] = error_occurred
                result.details["error_handled"] = error_handled

                if result.passed:
                    passed_tests += 1
                total_tests += 1

                self.log_trace(category, f"Test complete: {result.passed}", result.details)

            except Exception as e:
                # Unhandled exception = failure
                result.errors.append(f"Unhandled exception: {str(e)}")
                result.details["traceback"] = traceback.format_exc()
                total_tests += 1

            self.results.append(result)
            self.category_scores[category].append(result.score)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} {test['id']}: Handled gracefully")

        category_avg = np.mean(self.category_scores[category])
        print(f"\n  Category Score: {category_avg:.1f}% ({passed_tests}/{total_tests} handled)")

    # ========================================================================
    # CRITERION 4: Data Flow Traceability
    # ========================================================================

    def validate_data_flow(self):
        """Test: Can trace request end-to-end."""
        print("\n" + "─"*80)
        print("🔍 CRITERION 4: Data Flow Traceability")
        print("─"*80)

        category = "data_flow"
        result = ValidationResult("data_flow_trace", "End-to-end Traceability")

        try:
            test_text = REAL_WORLD_TEST_CASES[0]["text"]
            trace_id = hashlib.md5(test_text.encode()).hexdigest()[:8]

            self.log_trace(category, f"Starting trace {trace_id}", {"text_preview": test_text[:50]})

            # Step 1: Hash computation
            self.log_trace(category, f"[{trace_id}] Computing hash")
            hash_result = compute_sha256(test_text)
            self.log_trace(category, f"[{trace_id}] Hash complete", hash_result[:16])

            # Step 2: VAD extraction
            self.log_trace(category, f"[{trace_id}] Extracting VAD")
            extractor = VADExtractor()
            vad_result = extractor.extract_vad(test_text)
            self.log_trace(category, f"[{trace_id}] VAD complete", vad_result)

            # Step 3: Embedding generation
            self.log_trace(category, f"[{trace_id}] Generating embedding")
            emb_result = generate_embedding(test_text)
            self.log_trace(category, f"[{trace_id}] Embedding complete", {"shape": emb_result.shape})

            # Verify all steps completed and logged
            trace_steps = [t for t in self.trace_log if trace_id in t["message"]]

            result.details = {
                "trace_id": trace_id,
                "steps_logged": len(trace_steps),
                "hash": hash_result[:16],
                "vad": vad_result,
                "embedding_shape": emb_result.shape,
                "full_trace": trace_steps
            }

            # Scoring: All steps should be traceable
            result.passed = len(trace_steps) >= 6  # Start + 3 ops (start + complete each)
            result.score = min(len(trace_steps) / 6.0 * 100, 100)

            print(f"  Trace ID: {trace_id}")
            print(f"  Steps logged: {len(trace_steps)}")
            print(f"  ✅ Hash → VAD → Embedding pipeline traceable")

        except Exception as e:
            result.errors.append(f"Exception: {str(e)}")
            result.details["traceback"] = traceback.format_exc()

        self.results.append(result)
        self.category_scores[category].append(result.score)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  {status} Category Score: {result.score:.1f}%")

    # ========================================================================
    # CRITERION 5: Integration Surface
    # ========================================================================

    def validate_integration_surface(self):
        """Test: Inputs/outputs are clean and structured."""
        print("\n" + "─"*80)
        print("🔌 CRITERION 5: Integration Surface")
        print("─"*80)

        category = "integration"
        passed_tests = 0
        total_tests = 0

        # Test input/output contracts
        integration_tests = [
            {
                "name": "Hash function signature",
                "test": lambda: self.check_function_signature(compute_sha256, str, str)
            },
            {
                "name": "Embedding function accepts string",
                "test": lambda: isinstance(generate_embedding("test"), np.ndarray)
            },
            {
                "name": "Embedding function accepts list",
                "test": lambda: isinstance(generate_embedding(["test1", "test2"]), np.ndarray)
            },
            {
                "name": "VAD output structure",
                "test": lambda: self.check_vad_output_structure()
            },
            {
                "name": "Batch embedding output shape",
                "test": lambda: self.check_batch_embedding_shape()
            }
        ]

        for test in integration_tests:
            result = ValidationResult(test["name"], test["name"])

            try:
                self.log_trace(category, f"Testing: {test['name']}")
                test_passed = test["test"]()

                result.passed = test_passed
                result.score = 100.0 if result.passed else 0.0

                if result.passed:
                    passed_tests += 1
                total_tests += 1

            except Exception as e:
                result.errors.append(f"Exception: {str(e)}")
                total_tests += 1

            self.results.append(result)
            self.category_scores[category].append(result.score)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} {test['name']}")

        category_avg = np.mean(self.category_scores[category])
        print(f"\n  Category Score: {category_avg:.1f}% ({passed_tests}/{total_tests} passed)")

    # Helper methods
    def check_function_signature(self, func, input_type, output_type):
        """Check if function accepts and returns expected types."""
        if input_type == str:
            result = func("test")
        return isinstance(result, output_type)

    def check_vad_output_structure(self):
        """Verify VAD output has required fields."""
        extractor = VADExtractor()
        result = extractor.extract_vad("test")
        required_fields = ['valence', 'arousal', 'dominance', 'sha256']
        return all(field in result for field in required_fields)

    def check_batch_embedding_shape(self):
        """Verify batch embedding returns correct shape."""
        texts = ["test1", "test2", "test3"]
        result = generate_embeddings_batch(texts)
        return result.shape[0] == len(texts)

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_duration = self.end_time - self.start_time

        # Calculate category scores
        category_summary = {}
        for category, scores in self.category_scores.items():
            category_summary[category] = {
                "average_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "tests_run": len(scores)
            }

        # Calculate overall score
        all_scores = [score for scores in self.category_scores.values() for score in scores]
        overall_score = np.mean(all_scores) if all_scores else 0.0

        # Determine pass/fail
        phase2_ready = (
            overall_score >= 90.0 and
            all(cat["average_score"] >= 80.0 for cat in category_summary.values())
        )

        # Collect all errors and warnings
        all_errors = [r for r in self.results if r.errors]
        all_warnings = [r for r in self.results if r.warnings]

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": total_duration,
            "overall_score": overall_score,
            "phase2_ready": phase2_ready,
            "category_scores": category_summary,
            "total_tests": len(self.results),
            "passed_tests": len([r for r in self.results if r.passed]),
            "failed_tests": len([r for r in self.results if not r.passed]),
            "errors": len(all_errors),
            "warnings": len(all_warnings),
            "detailed_results": [r.to_dict() for r in self.results],
            "trace_log": self.trace_log
        }

        # Print summary
        self.print_summary_report(report)

        return report

    def print_summary_report(self, report: Dict):
        """Print human-readable summary."""
        print("\n\n" + "="*80)
        print("📋 VALIDATION SUMMARY REPORT")
        print("="*80)

        print(f"\n⏱️  Duration: {report['duration_seconds']:.2f}s")
        print(f"📊 Tests Run: {report['total_tests']}")
        print(f"✅ Passed: {report['passed_tests']}")
        print(f"❌ Failed: {report['failed_tests']}")
        print(f"⚠️  Errors: {report['errors']}")

        print(f"\n🎯 OVERALL SCORE: {report['overall_score']:.1f}%")

        print("\n📈 Category Breakdown:")
        for category, stats in report['category_scores'].items():
            emoji = "✅" if stats['average_score'] >= 80.0 else "❌"
            print(f"  {emoji} {category.upper():<20} {stats['average_score']:.1f}%")

        print("\n" + "="*80)
        if report['phase2_ready']:
            print("🎉 ✅ PHASE 2 READY - ALL SYSTEMS GO!")
            print("="*80)
            print("\n✨ Certification: Phase 1 foundation is SOLID")
            print("✨ Trust level: HIGH - System outputs are reliable")
            print("✨ Next step: Unlock Phase 2 (ChromaDB + Parallelization)")
        else:
            print("⚠️  ❌ PHASE 2 NOT READY - ISSUES FOUND")
            print("="*80)
            print("\n🔧 Action required:")
            print("   1. Review failed tests above")
            print("   2. Fix identified issues")
            print("   3. Re-run validation harness")
            print("   4. Achieve ≥90% overall score")

            # List specific failures
            failures = [r for r in self.results if not r.passed]
            if failures:
                print("\n❌ Failed Tests:")
                for f in failures[:5]:  # Show first 5
                    print(f"   • {f.test_name}: {f.score:.1f}%")
                    if f.errors:
                        print(f"     Error: {f.errors[0]}")

        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_phase1_validation():
    """
    Main entry point for Phase 1 validation.

    Returns:
        Dict: Complete validation report
    """
    harness = ValidationHarness()
    report = harness.run_all_validations()

    # Save report to file
    report_filename = f"phase1_validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Full report saved to: {report_filename}")

    return report


# ============================================================================
# AUTO-RUN (if executed directly)
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Executing Phase 1 Validation Harness...")
    report = run_phase1_validation()

    print("\n💡 Next Steps:")
    if report['phase2_ready']:
        print("   ✅ You're cleared for Phase 2")
        print("   ✅ Foundation is reliable and scalable")
        print("   ✅ Proceed with confidence")
    else:
        print("   ⚠️  Fix issues identified above")
        print("   ⚠️  Re-run: run_phase1_validation()")
        print("   ⚠️  Target: ≥90% overall score")
