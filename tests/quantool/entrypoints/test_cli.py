#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, "src")


def test_cli():
    print("=== CLI Test for GGUF ===")

    # Test registry
    from quantool.core.registry import QuantizerRegistry

    print("✓ Registry imported")

    # Test GGUF direct import
    from quantool.methods.llama_cpp.llama_cpp import GGUF

    print("✓ GGUF quantizer imported")
    print(f"Available quantizers: {QuantizerRegistry.list()}")

    # Test creating quantizer
    try:
        quantizer = QuantizerRegistry.create("gguf")
        print(f"✓ GGUF quantizer created: {type(quantizer).__name__}")
        print(f"Supported levels: {quantizer.supported_levels[:5]}...")  # Show first 5
    except Exception as e:
        print(f"✗ Failed to create GGUF quantizer: {e}")
        return False

    # Test CLI components
    try:
        from quantool.entrypoints.cli import quantize_step, validate_args_step

        print("✓ CLI functions imported")
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False

    print("=== CLI Test Completed Successfully ===")
    return True


if __name__ == "__main__":
    success = test_cli()
    sys.exit(0 if success else 1)
