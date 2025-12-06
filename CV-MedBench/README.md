# CV-MedBench: Clear-to-Vague Medical QA Benchmark

**CV-MedBench** is the first benchmark specifically designed to systematically evaluate the impact of **input ambiguity** on Medical Question Answering (QA) systems. It consists of carefully paired **clear** and **ambiguous (vague)** versions of real medical questions, enabling rigorous analysis of how LLMs and other QA models handle ambiguous inputs in the medical domain.

The benchmark is constructed by taking high-quality questions from three established medical QA datasets and creating corresponding ambiguous counterparts through controlled linguistic modifications while preserving the original correct answer.

## Source Datasets & Citations

CV-MedBench is built upon the following publicly available medical QA datasets:

- **MedQA** (USMLE-style questions in English)  
  Jin et al., [What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/abs/2009.13081).  

- **MedMCQA** (Indian medical entrance exam questions, multiple-choice)  
  Pal et al., [MedMCQA: A Large-scale Multi-subject Multi-choice Dataset for Medical Domain Question Answering](https://proceedings.mlr.press/v174/pal22a.html). 

- **MedExQA** (medical licensing examination questions across multiple languages)  
  Li et al., [MedExQA: Medical Licensing Examination Question-Answering Dataset across Multiple Languages](https://aclanthology.org/2024.bionlp-1.14/).  

## Dataset Structure

```
/CV-MedBench/
├── cv_medqa/
│   ├── train/
│   └── test/
├── cv_medmcqa/
│   ├── train/
│   └── test/
└── cv_medexqa/
    └── test/
```

Each split directory contains paired samples. For each original exam question, there are:

- A clear version  
- An ambiguity-induced version  

In addition, subsets with only one form of the question are stored as:

- `*_clear` (contains exclusively the clear question text)
- `*_vague` (contains exclusively the ambiguous question text)

The identifiers are aligned, meaning that the two versions of the same question carry the same `id` value.

---

## Data Schema

All subsets follow the same Arrow-based schema, compatible with Hugging Face `datasets`:

| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| **input** | The question text (either clear or ambiguous). | `string` |
| **output** | The answer aligned with the question. | `string` |
| **label** | Ambiguity flag: **0 = clear**, **1 = ambiguous**. | `int64` |
| **id** | Unique numerical identifier for paired queries. | `int64` |

**Schema Notes:**  
Samples sharing the same `id` represent the same underlying question, differing only in clarity.

---
For academic usage and experimental reporting involving this dataset, please cite our paper:
**“Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering.”**
