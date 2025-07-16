#!/usr/bin/env python3
"""
Test runner script for ScamAI detection pipeline tests

Usage:
    python test/test_runner.py                    # Run all tests
    python test/test_runner.py evaluation         # Run only evaluation tests
    python test/test_runner.py annotation         # Run only annotation tests
    python test/test_runner.py transcript         # Run only transcript tests
    python test/test_runner.py --verbose          # Run with verbose output
    python test/test_runner.py --coverage         # Run with coverage report
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_module=None, verbose=False, coverage=False):
    """Run pytest tests with specified options"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    
    if test_module:
        # Run specific test module
        test_files = {
            'evaluation': test_dir / 'test_evaluation.py',
            'annotation': test_dir / 'test_annotation.py', 
            'transcript': test_dir / 'test_transcript_generation.py'
        }
        
        if test_module not in test_files:
            print(f"Error: Unknown test module '{test_module}'")
            print(f"Available modules: {', '.join(test_files.keys())}")
            return 1
            
        cmd.append(str(test_files[test_module]))
    else:
        # Run all tests in test directory
        cmd.append(str(test_dir))
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add other useful pytest options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "-x",          # Stop on first failure
        "--strict-markers",  # Treat unknown markers as errors
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run ScamAI detection pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'module',
        nargs='?',
        choices=['evaluation', 'annotation', 'transcript'],
        help='Specific test module to run (default: run all tests)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests with verbose output'
    )
    
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Run tests with coverage report'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install test dependencies before running tests'
    )
    
    args = parser.parse_args()
    
    # Install test dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-cov", "pytest-asyncio"
        ])
        print("-" * 80)
    
    # Run tests
    exit_code = run_tests(
        test_module=args.module,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()