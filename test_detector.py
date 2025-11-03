#!/usr/bin/env python3
"""
Test script to verify PlagiarismDetector works correctly before running FastAPI
"""
import time
import sys


def test_imports():
    """Test that all required imports work"""
    print("=" * 60)
    print("TEST 1: Testing imports...")
    print("=" * 60)

    try:
        print("  Importing numpy...", end=" ")
        import numpy as np
        print("✓")

        print("  Importing sklearn...", end=" ")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓")

        print("  Importing requests...", end=" ")
        import requests
        print("✓")

        print("  Importing beautifulsoup4...", end=" ")
        from bs4 import BeautifulSoup
        print("✓")

        print("  Importing dotenv...", end=" ")
        from dotenv import load_dotenv
        print("✓")

        print("\n✅ All imports successful!\n")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nPlease install missing packages:")
        print("  pip install scikit-learn numpy requests beautifulsoup4 python-dotenv lxml")
        return False


def test_config():
    """Test Config initialization"""
    print("=" * 60)
    print("TEST 2: Testing Config initialization...")
    print("=" * 60)

    try:
        from plagiarism_detector import Config

        print("  Creating Config...", end=" ")
        config = Config()
        print("✓")

        print("\n  Configuration values:")
        print(f"    MAX_RESULTS_WEB: {config.MAX_RESULTS_WEB}")
        print(f"    MAX_RESULTS_SCHOLAR: {config.MAX_RESULTS_SCHOLAR}")
        print(f"    MAX_RESULTS_ARXIV: {config.MAX_RESULTS_ARXIV}")
        print(f"    SIMILARITY_THRESHOLD: {config.SIMILARITY_THRESHOLD}")
        print(f"    API_TIMEOUT: {config.API_TIMEOUT}")
        print(f"    SEARCHAPI_KEY: {'Set' if config.SEARCHAPI_KEY else 'Not set'}")

        print("\n✅ Config initialization successful!\n")
        return True, config

    except Exception as e:
        print(f"\n❌ Config initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_detector_init(config):
    """Test PlagiarismDetector initialization"""
    print("=" * 60)
    print("TEST 3: Testing PlagiarismDetector initialization...")
    print("=" * 60)

    try:
        from plagiarism_detector import PlagiarismDetector

        print("  Creating PlagiarismDetector...")
        start = time.time()
        detector = PlagiarismDetector(config)
        elapsed = time.time() - start

        print(f"\n✅ PlagiarismDetector initialized in {elapsed:.2f}s")
        print(f"   Knowledge base size: {len(detector.knowledge_base)} papers\n")
        return True, detector

    except Exception as e:
        print(f"\n❌ PlagiarismDetector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_detection(detector):
    """Test actual plagiarism detection"""
    print("=" * 60)
    print("TEST 4: Testing plagiarism detection...")
    print("=" * 60)

    test_text = (
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks that include an encoder "
        "and a decoder. We propose a new simple network architecture."
    )

    try:
        print(f"  Test text: {test_text[:80]}...")
        print("\n  Running detection (without web search)...")

        start = time.time()
        results = detector.detect_plagiarism(test_text, use_web_search=False)
        elapsed = time.time() - start

        if "error" in results:
            print(f"\n❌ Detection returned error: {results['error']}")
            return False

        print(f"\n✅ Detection completed in {elapsed:.2f}s")
        print(f"\n  Results:")
        print(f"    Overall Score: {results['overall_score'] * 100:.1f}%")
        print(f"    Verdict: {results['verdict']}")
        print(f"    Total Matches: {results['total_matches']}")
        print(f"    Word Count: {results['word_count']}")

        if results['matches']:
            print(f"\n  Top match:")
            top = results['matches'][0]
            print(f"    Title: {top['title']}")
            print(f"    Source: {top['source']}")
            print(f"    Similarity: {top['similarity'] * 100:.1f}%")
            print(f"    Details:")
            for key, val in top['details'].items():
                print(f"      {key}: {val * 100:.1f}%")

        print("\n✅ Detection test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("PLAGIARISM DETECTOR TEST SUITE")
    print("=" * 60 + "\n")

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Tests failed at import stage")
        sys.exit(1)

    # Test 2: Config
    success, config = test_config()
    if not success:
        print("\n❌ Tests failed at config stage")
        sys.exit(1)

    # Test 3: Detector initialization
    success, detector = test_detector_init(config)
    if not success:
        print("\n❌ Tests failed at detector initialization stage")
        sys.exit(1)

    # Test 4: Detection
    if not test_detection(detector):
        print("\n❌ Tests failed at detection stage")
        sys.exit(1)

    # All tests passed
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour detector is working correctly. You can now run FastAPI:")
    print("  python main.py")
    print("  or")
    print("  uvicorn main:app --reload")
    print()


if __name__ == "__main__":
    main()