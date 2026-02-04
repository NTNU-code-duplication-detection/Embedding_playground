# Pre-commit Setup Guide

This guide will help you set up pre-commit hooks for this project.

## Prerequisites

- Python 3.x installed
- Git installed
## Setup Instructions

### 1. Create Virtual Environment (if not exists)

**Windows PowerShell:**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pre-commit
```


### 3. Install Pre-commit Hooks
```bash
pre-commit clean
pre-commit install
pre-commit install --hook-type commit-msg
```

## Usage

### Commit Message Format

All commits must follow this format: `TYPE: description`

**Allowed types:**
- `FEAT`: New feature
- `FIX`: Bug fix
- `DOCS`: Documentation
- `REFACTOR`: Code refactoring
- `CHORE`: Maintenance (build configs, dependencies etc)
- `TEST`: Adding or updating tests
- `MERGE`: Merging branches 

**Examples:**

**Valid commits:**
```Powershel
git commit -m "FEAT: add user authentication"
git commit -m "FIX: resolve login bug"
git commit -m "DOCS: update README"
```

**Invalid commits:**
```bash
git commit -m "fixed some stuff"
git commit -m "feat: add feature"  # lowercase not allowed
git commit -m "UPDATE: something"  # wrong type
```

### Testing the Setup
```bash
# This will FAIL (wrong format)
git commit --allow-empty -m "fixed some stuff"

# This will PASS
git commit --allow-empty -m "FIX: fixed some stuff"
```

### Run Pre-commit on All Files
```bash
pre-commit run --all-files
```

## What Gets Checked

When you commit, the following checks run automatically:

1. **Commit Message Format** - Validates conventional commit format
2. **PyLint** - Runs on all `.py` files

## Troubleshooting

**If hooks don't run:**
```bash
pre-commit clean
pre-commit install
pre-commit install --hook-type commit-msg
```

**Skip hooks (emergency only):**
```bash
git commit --no-verify -m "CHORE: emergency fix"
```