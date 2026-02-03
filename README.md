# Code Model Embedding Pipeline

This pipeline focuses on type-1 and type-2 code duplication using code embedding models.
Duplication is classified using spatial closeness in the embedding-space.

## Requirements
This project uses python as the backend
- python3
- pip


## Installation
Run these command in your shell
```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Datasets

This project uses the following external dataset as a Git submodule:

- **sourcecodeplagiarismdataset**
  - Repository: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
  - Purpose: Source code plagiarism benchmark dataset
  - License: Apache License
  - Location: `datasets/sourcecodeplagiarismdataset`

### Cloning

To clone this repository with all datasets:

```bash
git clone --recurse-submodules <this-repo-url>
```