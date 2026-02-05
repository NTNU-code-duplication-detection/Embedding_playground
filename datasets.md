# Datasets
This project uses four external datasets: one as a Git submodule, one large dataset from OneDrive, and one from HuggingFace.

## 1 & 2. Git Submodule Dataset
First submodule
- **Name:** sourcecodeplagiarismdataset
- **Repository:** https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
- **Purpose:** Source code plagiarism benchmark dataset
- **Licence:** Apache License 2.0
- **Location in project:** `datasets/sourcecodeplagiarismdataset`

Second submodule
Created by us\
[Link](https://github.com/NTNU-code-duplication-detection/code-clone-dataset)

### Cloning the Repository with Submodules
To clone this repository including all  Git submodules:

```bash
git clone --recursive-submodules https://github.com/NTNU-code-duplication-detection/dataset_loader
```

If you already cloned the repo without submodules, you can initalize them:
```bash
git submodule update --init --recursive
```

## 3. Large Dataset: Pure IJaDataset + BigCloneBench Samples
- **Source:** [BigCloneBench Version 2](https://github.com/clonebench/BigCloneBench?tab=readme-ov-file#bigclonebench-version-2-use-this-version)
- **Direct Download Link:** [OneDrive](https://1drv.ms/u/s!AhXbM6MKt_yLj_tk29GJnc9BKoIvCg?e=oVTVJm)
- **Purpose:** Large dataset of code samples for clone detection
- **Note:** This dataset is **very large** and should be downloaded and extracted in place, because moving it afterward may be slow or fail.

### Recommended Folder Structure
```text
~/data/
    bigclonebench/                      # Large dataset from OneDrive

project/                                # Git repository
    data/
        sourcecodeplagiarismdataset/    # Submodule
```

### Extraction Instructions
#### macOS / Linux
```bash
# Create datasets folder
mkdir -p ~/datasets/bigclonebench

# Extract the dataset (assuming downloaded to ~/Downloads/dataset.tar.gz)
tar -xf ~/Downloads/dataset.tar.gz -C ~/datasets/bigclonebench
```
NOTE: Make sure to extract the dataset directly into ~/datasets/to avoid issues with moving large files later.

### Custom Dataset Location
By default, the BigCloneBench dataset is expected at `~/datasets/bigclonebench`.
This can be overridden by setting the `BIGCLONEBENCH_ROOT` environment variable.
```bash
export BIGCLONEBENCH_ROOT=/mnt/ssd/bigclonebench
```


## 4. CodeXGlue from Google/Microsoft/Madlag
- **Source:** [Hugging Face - CodeXGlue Clone detection](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench)
- **Purpose:** Provides labeled pairs of Java methods from BigCloneBench, for training and evaluating code clone detection models. Each pair has a binary label indicating whether the two methods are clones/plagiarized (1) or not (0).
- **Format:** JSON/CSV-style dataset with fields including
```
{
    "func1": "      # Full text of frist function 
    "func2": "      # Full text of second function
    "id": 0,        # Index of sample
    "id1": 2381663, # First function ID
    "id2": 4458076, # Second function ID
    "label": false  # true of functions are not eqivalent, false otherwise
}
```
- **License:** Computational Use of Data Agreement (C-UDA) License.
- **Location in project:** This dataset is **downloaded on demand** via Huggingface Datasets Library
