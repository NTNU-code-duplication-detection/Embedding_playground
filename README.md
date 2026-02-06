# Dataset Loader

This project focuses on the delivery and preprocessing of data from several datasets. See `datasets.md` for which and how to install.

## Requirements
This project uses python as the backend
- `python3`
- `pip`


## Installation
Run these commands in your shell
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Preprocessing Tools

### Comment Stripper

`preprocessing/comment_stripper.py`

Strips all comments and Javadoc from Java source files. This is used as a preprocessing step before generating embeddings, so that comments don't influence similarity comparisons between code files.

#### What it removes
- Single-line comments: `// ...`
- Multi-line comments: `/* ... */`
- Javadoc comments: `/** ... */`
- Inline comments at the end of code lines

#### What it preserves
- String literals (e.g. `"http://example.com"` is not treated as a comment)
- Character literals (e.g. `'/'` is not treated as a comment)
- All actual code and structure

#### How it works

The stripper uses a character-by-character state machine parser rather than regex. This avoids false positives where comment-like syntax appears inside string or character literals. It walks through the source tracking whether the current position is inside a string literal, character literal, or regular code, and only strips content that is actually a comment.

After stripping, it also:
1. Removes trailing whitespace from each line
2. Collapses consecutive blank lines (left behind by removed comment blocks) down to a single blank line

#### CLI usage

Takes an input directory and an output directory. Recursively finds all `.java` files in the input, strips comments, and writes cleaned copies to the output — preserving the original folder structure. Original files are never modified.

```bash
python -m preprocessing.comment_stripper \
    --input ../code-clone-dataset/dataset \
    --output ./output/stripped
```

#### Library usage

The functions can also be imported directly for use in other scripts or pipelines:

```python
from preprocessing.comment_stripper import process_file, process_directory
from pathlib import Path

# Strip a single source string
cleaned = process_file(java_source_code)

# Process an entire directory tree
stats = process_directory(Path("input/"), Path("output/"))
print(f"Processed {stats['files_processed']} files")
```