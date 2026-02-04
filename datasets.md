# Datasets
This project uses two external datasets: one as a Git submodule, and one large dataset from OneDrive.

## 1. Git Submodule Dataset
- **Name:** sourcecodeplagiarismdataset
- **Repository:** https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
- **Purpose:** Source code plagiarism benchmark dataset
- **Licence:** Apache License 2.0
- **Location in project:** `datasets/sourcecodeplagiarismdataset`

### Cloning the Repository with Submodules
To clone this repository including all Git submodules:

```bash
git clone --recursive-submodules <this-repo-url>
```

If you already cloned the repo without submodules, you can initalize them:
```bash
git submodule update --init --recursive
```

## 2. Large Dataset: Pure IJaDataset + BigCloneBench Samples
- **Source:** [BigCloneBench Version 2](https://github.com/clonebench/BigCloneBench?tab=readme-ov-file#bigclonebench-version-2-use-this-version)
- **Direct Download Link:** [OneDrive](https://1drv.ms/u/s!AhXbM6MKt_yLj_tk29GJnc9BKoIvCg?e=oVTVJm)
- **Purpose:** Large dataset of code samples for clone detection
- **Note:** This dataset is **very large** and should be downloaded and extracted in place, because moving it afterward may be slow or fail.

### Recommended Folder Structure
```text
~/datasets/bigclonebench                        # Large dataset from OneDrive
project/                                    # Git repository
    datasets/sourcecodeplagiarismdataset/   # Submodule
```

### Extraction Instructions
#### Ubuntu/Linux
```bash
# Create datasets folder
mkdir -p ~/datasets/bigclonebench

# Extract the dataset (assuming downloaded to ~/Downloads/dataset.tar.gz)
tar -xf ~/Downloads/dataset.tar.gz -C ~/datasets/bigclonebench
```

#### macOS
```bash
# Create datasets folder
mkdir -p ~/datasets/bigclonebench

# Extract the dataset (assuming downloaded to ~/Downloads/dataset.zip)
unzip ~/Downloads/dataset.tar.gz -d ~/dataset/bigclonebench
```
NOTE: Make sure to extract the dataset directly into ~/datasets/to avoid issues with moving large files later.

