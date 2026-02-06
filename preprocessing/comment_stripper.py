"""
Strips comments and Javadoc from Java source files.

Recursively traverses an input directory, finds all .java files,
removes comments (single-line, multi-line, Javadoc), and writes
cleaned files to an output directory preserving folder structure.

Usage:
    python -m preprocessing.comment_stripper \\
        --input ../code-clone-dataset/dataset \\
        --output ./output/stripped
"""

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def _consume_literal(source: str, i: int, quote: str) -> tuple[list[str], int]:
    """Consume a string or character literal, returning chars and new index."""
    length = len(source)
    chars: list[str] = [quote]
    i += 1
    while i < length and source[i] != quote:
        if source[i] == '\\':
            chars.append(source[i])
            i += 1
        if i < length:
            chars.append(source[i])
            i += 1
    if i < length:
        chars.append(source[i])
        i += 1
    return chars, i


def strip_java_comments(source: str) -> str:
    """
    Remove all comments from Java source code using a state-machine parser.

    Handles single-line (//), multi-line (/* */), and Javadoc (/** */)
    comments while preserving string and character literals.

    Args:
        source: Raw Java source code.

    Returns:
        Source code with all comments removed.
    """
    result: list[str] = []
    i = 0
    length = len(source)

    while i < length:
        char = source[i]

        # String or character literal — pass through unchanged
        if char in ('"', "'"):
            chars, i = _consume_literal(source, i, char)
            result.extend(chars)

        # Single-line comment: skip to end of line
        elif char == '/' and i + 1 < length and source[i + 1] == '/':
            i += 2
            while i < length and source[i] != '\n':
                i += 1

        # Multi-line or Javadoc comment: skip to closing */
        elif char == '/' and i + 1 < length and source[i + 1] == '*':
            i += 2
            while i + 1 < length and not (source[i] == '*' and source[i + 1] == '/'):
                i += 1
            i += 2  # skip closing */

        else:
            result.append(char)
            i += 1

    return ''.join(result)


def strip_trailing_whitespace(source: str) -> str:
    """Remove trailing whitespace from each line."""
    lines = source.split('\n')
    return '\n'.join(line.rstrip() for line in lines)


def clean_blank_lines(source: str) -> str:
    """Collapse consecutive blank lines down to a single blank line."""
    return re.sub(r'\n{3,}', '\n\n', source)


def process_file(source: str) -> str:
    """Full preprocessing pipeline: strip comments, clean up whitespace."""
    result = strip_java_comments(source)
    # Strip trailing whitespace first so comment-only lines become truly empty,
    # then collapse consecutive blank lines
    result = strip_trailing_whitespace(result)
    result = clean_blank_lines(result)
    return result


def process_directory(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Recursively find .java files, strip comments, write to output directory.

    Args:
        input_dir: Root directory to search for .java files.
        output_dir: Root directory to write stripped files to.

    Returns:
        Dict with processing statistics.
    """
    stats = {"files_processed": 0, "files_skipped": 0, "errors": 0}

    java_files = sorted(input_dir.rglob("*.java"))
    if not java_files:
        log.warning("No .java files found in %s", input_dir)
        return stats

    log.info("Found %d .java files in %s", len(java_files), input_dir)

    for java_file in java_files:
        relative = java_file.relative_to(input_dir)
        output_file = output_dir / relative

        try:
            source = java_file.read_text(encoding="utf-8")
            stripped = process_file(source)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(stripped, encoding="utf-8")

            stats["files_processed"] += 1
        except Exception as err:  # pylint: disable=broad-except
            log.error("Failed to process %s: %s", java_file, err)
            stats["errors"] += 1

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Strip comments and Javadoc from Java source files.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing .java files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for stripped files.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_dir.is_dir():
        log.error("Input path is not a directory: %s", input_dir)
        return

    if input_dir == output_dir:
        log.error("Input and output directories must be different.")
        return

    # Guard against output being a subdirectory of input, which would
    # cause the tool to process its own output files
    if output_dir.is_relative_to(input_dir):
        log.error("Output directory cannot be inside the input directory.")
        return

    log.info("Input:  %s", input_dir)
    log.info("Output: %s", output_dir)

    if output_dir.exists():
        existing = list(output_dir.rglob("*.java"))
        if existing:
            log.warning(
                "Output directory already contains %d .java files. "
                "Existing files will be overwritten.",
                len(existing),
            )

    stats = process_directory(input_dir, output_dir)

    log.info("Done. Processed: %d, Errors: %d",
             stats["files_processed"], stats["errors"])


if __name__ == "__main__":
    main()
