# cfDNA Fragmentomics for Early Detection of Urological Cancers

This repository contains the code and scripts used in the study investigating the diagnostic potential of **circulating free DNA (cfDNA) fragmentomics** for the early detection of bladder urothelial carcinoma (BLCA), prostate adenocarcinoma (PRAD), and clear cell renal cell carcinoma (ccRCC). Our work utilizes **low-coverage whole-genome sequencing (lcWGS)** data combined with **machine learning (ML)** algorithms to identify distinctive cfDNA fragmentation patterns, end motifs (EDMs), and breakpoint motifs (BPMs) associated with these cancers.

---

## Project Overview

Early diagnosis of urological cancers is critical for improving patient outcomes. Traditional screening methods often suffer from limitations such as low specificity, invasiveness, or lack of established biomarkers. This project proposes a novel, non-invasive liquid biopsy approach based on cfDNA fragmentomic analysis. We aim to develop robust machine learning models capable of distinguishing early-stage urological cancer patients from non-cancer controls, including individuals with benign prostatic hyperplasia (BPH) and healthy individuals.

---

## Key Features

* **Comprehensive cfDNA Fragmentomic Analysis:** Extracts and analyzes a wide range of cfDNA features including Fragment Size Ratio (FSR), Fragment Size Distribution (FSD), End Motifs (EDMs), and Breakpoint Motifs (BPMs) from lcWGS data.
* **Machine Learning Model Development:** Implements and evaluates multiple machine learning algorithms (Logistic Regression, SVM, Random Forest, XGBoost, Stacking) for classification.
* **Robust Feature Selection:** Employs a two-step feature selection strategy involving T-tests and Recursive Feature Elimination with Cross-Validation (RFECV) guided by SHAP values to identify optimal diagnostic features.
* **Pan-Cancer and Cancer-Specific Models:** Develops models for individual cancer types (BLCA, PRAD, ccRCC) as well as a pan-cancer model for general urological tumor detection.
* **Two-Tiered Screening Strategy:** Proposes and evaluates a cost-effective, sequential screening approach starting with a pan-cancer screen followed by cancer-specific differential diagnosis.
* **Model Interpretability:** Utilizes SHAP (SHapley Additive exPlanations) to interpret model predictions and highlight the contribution of key cfDNA features.

---

## Repository Structure

```

.
├── src/
│   ├── data\_preprocessing/     \# Scripts for raw data processing (e.g., alignment, QC)
│   ├── feature\_extraction/     \# Scripts for extracting FSR, FSD, EDM, BPM features
│   ├── feature\_selection/      \# Scripts for T-test, RFECV, and SHAP-based feature selection
│   ├── model\_training/         \# Scripts for training and hyperparameter optimization of ML models
│   ├── model\_evaluation/       \# Scripts for evaluating model performance (AUC, DCA, sensitivity/specificity)
│   └── utils/                  \# Helper functions and common utilities
├── data/
│   ├── raw/                    \# Placeholder for raw sequencing data (e.g., FASTQ files - typically large, not directly hosted)
│   ├── processed/              \# Processed cfDNA feature matrices (e.g., CSV, Parquet files)
│   └── interim/                \# Intermediate data files generated during processing
├── notebooks/                  \# Jupyter notebooks for exploratory data analysis, visualization, and results presentation
│   ├── EDA.ipynb
│   ├── Model\_Performance\_Analysis.ipynb
│   └── Feature\_Importance\_SHAP.ipynb
├── config/                     \# Configuration files (e.g., parameters for feature extraction, model hyperparameters)
├── results/                    \# Directory for storing model outputs, performance metrics, plots, and tables
│   ├── models/                 \# Saved trained models
│   ├── plots/                  \# Generated figures (e.g., ROC curves, DCA plots, waterfall plots, SHAP plots)
│   └── tables/                 \# Performance tables, feature lists
├── environment.yml             \# Conda environment file for reproducible environment setup
├── requirements.txt            \# Python dependencies for pip installation
└── README.md                   \# This file

````

---

## Getting Started

### Prerequisites

* **Python 3.x**
* **Conda** (recommended for environment management) or **pip**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/cfDNA-Urological-Cancers.git](https://github.com/YourGitHubUsername/cfDNA-Urological-Cancers.git)
    cd cfDNA-Urological-Cancers
    ```
2.  **Set up the Conda environment (recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate cfdna_uro_env
    ```
3.  **Alternatively, install dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Availability

Due to privacy concerns and file size, raw sequencing data (FASTQ files) cannot be directly hosted in this repository. However, the scripts in `src/data_preprocessing/` are designed to handle such data.

**Processed feature matrices** necessary for reproducing the analysis and model training will be made available in the `data/processed/` directory or linked via a data repository if too large for GitHub. Please refer to `data/processed/README.md` (to be created) for details on how to access or generate the feature matrices.

### Usage

The analysis workflow can be generally followed by executing the scripts in the `src/` directory in sequence, or by running the Jupyter notebooks in `notebooks/`.

1.  **Data Preparation:** (Assuming raw data is accessible as described above)
    * `src/data_preprocessing/run_alignment_and_qc.sh` (example script for bioinformatics pipeline)
    * `src/feature_extraction/extract_fragment_features.py`
    * `src/feature_extraction/extract_motif_features.py`
    * *Note: These scripts may require significant computational resources and specific bioinformatics tools not directly included in the Python environment.*

2.  **Feature Selection:**
    * `src/feature_selection/run_t_tests.py`
    * `src/feature_selection/run_rfecv_shap.py`

3.  **Model Training and Evaluation:**
    * `src/model_training/train_all_models.py`
    * `src/model_evaluation/evaluate_performance.py`
    * `src/model_evaluation/generate_dca_plots.py`
    * `src/model_evaluation/generate_shap_plots.py`
    * `src/model_evaluation/generate_waterfall_plots.py`

Please refer to the individual script files and Jupyter notebooks for detailed usage instructions and parameters.

---

## Results and Findings

The key results of this study, including diagnostic performance (AUCs, sensitivity, specificity), decision curve analyses, and SHAP plots illustrating feature contributions, are discussed in the accompanying manuscript and visualized in the `results/plots/` directory.

**Highlights:**
* Achieved high AUCs: BLCA (96%), ccRCC (99%), PRAD (92%), and pan-cancer (89%).
* Identified 6-bp EDMs and BPMs as critical discriminating features.
* Proposed a two-tier screening strategy offering enhanced efficiency and cost-effectiveness.

---

## Contribution

We welcome contributions! If you find issues or have suggestions for improvements, please open an issue or submit a pull request.

---

## Citation

If you use this code or data in your research, please cite our corresponding publication (details to be updated upon publication):

````

[Your Publication Details Here]

```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or inquiries, please contact:
[Your Name/Corresponding Author Name]
[Your Email Address]
```
