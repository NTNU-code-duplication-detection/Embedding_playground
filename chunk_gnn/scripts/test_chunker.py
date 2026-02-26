"""
Quick test for the TreeSitterChunker.

Run on IDUN (or anywhere with tree-sitter-java installed) to verify
the chunker works on BCB-style Java functions.

Usage:
    python scripts/test_chunker.py
    python scripts/test_chunker.py --bcb_root /path/to/dataset_bigclonebench
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.chunker import TreeSitterChunker, ChunkKind


# Sample BCB-style Java functions (bare methods, no class wrapper)
SAMPLE_FUNCTIONS = [
    # Simple method with variable declarations and return
    """
    public static int add(int a, int b) {
        int result = a + b;
        return result;
    }
    """,
    # Method with if-else
    """
    public static int max(int a, int b) {
        if (a > b) {
            return a;
        } else {
            return b;
        }
    }
    """,
    # Method with for loop and if
    """
    public static boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
    """,
    # Method with enhanced for loop (for-each)
    """
    public static int sum(int[] arr) {
        int total = 0;
        for (int x : arr) {
            total += x;
        }
        return total;
    }
    """,
    # Method with try-catch
    """
    public static int parseInt(String s) {
        int result = 0;
        try {
            result = Integer.parseInt(s);
        } catch (NumberFormatException e) {
            System.out.println("Invalid number: " + s);
        }
        return result;
    }
    """,
    # Method with while loop
    """
    public static int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    """,
    # Empty method
    """
    public void doNothing() {
    }
    """,
    # Single return statement
    """
    public int getX() {
        return this.x;
    }
    """,
]


def test_sample_functions():
    """Test chunker on built-in sample functions."""
    chunker = TreeSitterChunker()
    passed = 0
    failed = 0

    for i, source in enumerate(SAMPLE_FUNCTIONS):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {source.strip().split(chr(10))[0][:60]}...")
        print(f"{'='*60}")

        chunks = chunker.chunk_function(source)

        if chunks is None:
            print("  Result: None (no chunks extracted)")
            # Empty method and single-return are expected to return None or few chunks
            if "doNothing" in source:
                print("  [OK] Expected: empty method produces no chunks")
                passed += 1
            else:
                print("  [WARN] Unexpected None result")
                failed += 1
            continue

        print(f"  Chunks: {len(chunks)}")
        straight = sum(1 for c in chunks if c.kind == ChunkKind.STRAIGHT)
        control = sum(1 for c in chunks if c.kind == ChunkKind.CONTROL)
        print(f"  STRAIGHT: {straight}, CONTROL: {control}")

        for j, chunk in enumerate(chunks):
            parent_str = f"parent={chunk.parent_index}" if chunk.parent_index is not None else "root"
            print(
                f"  [{j}] {chunk.kind.value:8s} depth={chunk.depth} "
                f"lines={chunk.start_line}-{chunk.end_line} {parent_str}"
            )
            # Indent the text for readability
            text_preview = chunk.text[:80].replace("\n", "\\n")
            print(f"       text: {text_preview!r}")

        # Verify parent indices are valid
        for j, chunk in enumerate(chunks):
            if chunk.parent_index is not None:
                assert 0 <= chunk.parent_index < j, (
                    f"Chunk {j} has invalid parent_index {chunk.parent_index}"
                )

        # Verify sequential ordering (start lines should be non-decreasing)
        for j in range(1, len(chunks)):
            assert chunks[j].start_line >= chunks[j-1].start_line, (
                f"Chunks not in order: chunk {j} starts at line "
                f"{chunks[j].start_line} before chunk {j-1} at "
                f"{chunks[j-1].start_line}"
            )

        print("  [OK] All assertions passed")
        passed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(SAMPLE_FUNCTIONS)}")
    print(f"{'='*60}")
    return failed == 0


def test_bcb_files(bcb_root: str, num_samples: int = 20):
    """Test chunker on actual BCB files."""
    from chunk_gnn.data.bcb_loader import BCBLoader

    print(f"\n{'='*60}")
    print(f"TESTING ON BCB FILES ({num_samples} random samples)")
    print(f"{'='*60}")

    loader = BCBLoader(bcb_root=bcb_root)
    available, missing = loader.check_file_availability()
    print(f"Available files: {len(available)}, Missing: {len(missing)}")

    chunker = TreeSitterChunker()
    available_list = list(available)
    random.seed(42)
    sample_ids = random.sample(
        available_list, min(num_samples, len(available_list))
    )

    results = {"success": 0, "empty": 0, "failed": 0}
    chunk_counts = []

    for func_id in sample_ids:
        source = loader.load_function_source(func_id)
        if source is None:
            print(f"  {func_id}: could not read file")
            results["failed"] += 1
            continue

        chunks = chunker.chunk_function(source)
        if chunks is None:
            print(f"  {func_id}: no chunks (empty/unparseable, {len(source.strip().splitlines())} lines)")
            results["empty"] += 1
        else:
            s = sum(1 for c in chunks if c.kind == ChunkKind.STRAIGHT)
            c = sum(1 for c in chunks if c.kind == ChunkKind.CONTROL)
            print(f"  {func_id}: {len(chunks)} chunks (S={s}, C={c})")
            results["success"] += 1
            chunk_counts.append(len(chunks))

    print(f"\nBCB Results: {results}")
    if chunk_counts:
        avg = sum(chunk_counts) / len(chunk_counts)
        print(f"Chunk count stats: min={min(chunk_counts)}, max={max(chunk_counts)}, avg={avg:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Test the TreeSitterChunker")
    parser.add_argument(
        "--bcb_root", type=str, default=None,
        help="Path to BCB data dir (optional, for testing on real BCB files)",
    )
    args = parser.parse_args()

    all_ok = test_sample_functions()

    if args.bcb_root:
        test_bcb_files(args.bcb_root)

    if all_ok:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
