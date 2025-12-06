# 🩺 AU-Med: Aleatoric Uncertainty Quantification for Safe Medical QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/IINemo/lm-polygraph/blob/master/LICENSE.md)

This repository contains the code and implementation for the **AU-Probe** module, developed to address critical safety risks posed by **ambiguous user queries** in Medical Question Answering (Medical QA). The core idea is to formalize input ambiguity as **Aleatoric Uncertainty (AU)**, and quantify it efficiently using the Representation Engineering technique.

This work is built upon the open-source [LM-Polygraph](https://github.com/IINemo/lm-polygraph) library (MIT License). All novel methodological contributions (AU-Probe, CV-MedBench dataset, and the AU-Guided Clarify-Before-Answer framework) are original to this project.


## 1. Setup and Installation

### 1.1 Environment Setup

```bash
# Create and activate a clean environment (recommended)
conda create -n aumed python=3.11 -y
conda activate aumed

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Data and Pre-trained Probes

Please ensure the following files are placed in their corresponding directories:

- **CV-MedBench** dataset  
  → Place all dataset files in: `./CV-MedBench/`

- **Pre-trained AU-Probe weights** (linear probes for each layer and supported model)  
  → Place all probe weight files in: `./hidden_states/`

These paths are directly referenced by the Hydra configuration files in `./examples/configs/`. 

-----

## 3\. Running Model Evaluation

The evaluation is executed using the pre-configured Hydra YAML files and the respective Bash scripts located in the `run/` directory.

### 3.1 Command Structure

The bash scripts run the main evaluation program (`scripts/polygraph_eval`) and overwrite the configuration (model, dataset, layer) to perform the experiment.

**Execution Command:**

```bash
# Example: Run the evaluation for the MedQA subset on BioMistral-7B
bash run/medqa.sh 
```

### 3.2 Output Generation

Successfully running the script will generate **model outputs, estimation scores, and verbose batch logs** in your defined directory structure:

`workdir/output/qa/${ModelPath}/${DatasetName}/${Tag}/${Time}/polygraph_eval.log`

The log file contains the model's generated answer, estimator scores, and the ambiguity label for each batch.

-----

## 4\. Analysis and Metric Calculation

The `analyze.py` script automatically processes the raw log files to extract the necessary metrics for comparing UQ methods.

### 4.1 Script Execution and Purpose

This command aggregates data across all runs, calculates performance metrics, and outputs summarized CSV files.

```bash
# Execute the analysis script from the project root directory
python scripts/analyze.py
```

### 4.2 Metrics Calculated

The analysis script processes the per-batch data and calculates the following final metrics for each (Model, Dataset, Estimator) combination:

| Metric | Category | Goal |
| :--- | :--- | :--- |
| **AUROC** | Discrimination | **Higher is better** |
| **ECE** | Calibration | **Lower is better** |
| **Brier Score** | Quality/Sharpness | **Lower is better** |

### 4.3 CSV Output

The results are saved in CSV files, structured by model and dataset, at a path like:

`workdir/output/DatasetName/ModelName/ModelName_DatasetName_metrics.csv`

The CSV output will include columns for the final calculated metrics: `Timestamp`, `Estimator`, `AUROC`, `ECE`, and `Brier`.
