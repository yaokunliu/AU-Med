# 🩺 AU-Med: Aleatoric Uncertainty Quantification for Safe Medical QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/IINemo/lm-polygraph/blob/master/LICENSE.md)

This repository contains the official code implementation for the paper:

**Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering**,  
in the Proceedings of The Web Conference (WWW) 2026.

The project introduces **AU-Probe**, a lightweight module for quantifying aleatoric uncertainty arising from ambiguous user inputs in Medical Question Answering (Medical QA). By modeling input ambiguity as aleatoric uncertainty and estimating it directly from internal representations of large language models, this work enables early ambiguity detection and supports safer, clarification-aware QA systems.

This codebase is built upon the open-source [LM-Polygraph](https://github.com/IINemo/lm-polygraph) library (MIT License). All methodological contributions presented in the paper—including **AU-Probe**, the **CV-MedBench** dataset, and the **AU-Guided Clarify-Before-Answer** framework—are implemented and released as part of this repository.

## Citation

If you use this code in academic work, please cite:

**Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering**  
Yaokun Liu *et al.*, The Web Conference (WWW) 2026.

---

## Dataset: CV-MedBench

The experiments in this repository are conducted on **CV-MedBench**, a clear-to-vague medical question answering benchmark designed to study input ambiguity and aleatoric uncertainty in Medical QA.

The dataset is publicly available on Hugging Face:

👉 **https://huggingface.co/datasets/yaokunl/CV-MedBench**

CV-MedBench contains paired **clear** and **ambiguous** versions of real medical exam questions derived from MedQA, MedMCQA, and MedExQA, with aligned identifiers to support controlled evaluation under different clarity conditions.  

---

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
