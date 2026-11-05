[![DOI](https://zenodo.org/badge/1445358389.svg)](https://doi.org/10.5281/zenodo.23941358)

# VariantExplainAI

### An Explainable Deep Learning Framework for Genomic Variant Pathogenicity Prediction

## 1. Overview

**VariantExplainAI** is an explainable deep learning framework designed
for genomic variant pathogenicity prediction using ClinVar data. The
framework integrates allele-specific k-mer embeddings, CNN-based motif
extraction, and sparse Transformer architectures to capture local
genomic patterns and long-range contextual dependencies for variant
classification.

To improve model transparency and interpretability, VariantExplainAI
incorporates Integrated Gradients (IG), perturbation-based analysis, and
attention rollout mechanisms. Explanation-guided optimization is used to
improve the consistency and reliability of model explanations while
supporting pathogenicity prediction.

The framework provides an integrated pipeline covering genomic data
preparation, model training, baseline comparison, explainability
assessment, statistical evaluation, and visualization.

## 2. Key Features

-   **Allele-Specific K-mer Embeddings:** Represents reference and
    alternate genomic alleles using sequence-based feature embeddings.
-   **CNN-Based Motif Extraction:** Captures local genomic patterns and
    discriminative sequence motifs.
-   **Sparse Transformer Architecture:** Models contextual dependencies
    while reducing unnecessary attention computation.
-   **Integrated Gradients:** Identifies important genomic features
    contributing to pathogenicity predictions.
-   **Perturbation-Based Explainability:** Evaluates prediction
    sensitivity through controlled input modifications.
-   **Attention Rollout:** Provides contextual interpretation of
    attention-based feature interactions.
-   **Explanation-Guided Optimization:** Incorporates
    explanation-related objectives into model optimization.
-   **Comprehensive Evaluation:** Supports predictive performance,
    temporal analysis, statistical assessment, and explainability
    evaluation.
-   **Ablation Studies:** Evaluates the contribution of individual
    architectural components.
-   **Baseline Comparison:** Facilitates comparative evaluation against
    alternative prediction approaches.

## 3. Proposed Architecture

The VariantExplainAI framework consists of the following major
components:

1.  **Genomic Data Preparation:** Processes ClinVar variant records and
    prepares reference and alternate allele sequences.
2.  **Allele-Specific Representation:** Generates k-mer-based embeddings
    to represent genomic sequence variations.
3.  **Local Motif Extraction:** Uses convolutional neural networks
    (CNNs) to extract local sequence characteristics.
4.  **Contextual Feature Learning:** Applies sparse Transformer
    attention to capture contextual genomic dependencies.
5.  **Pathogenicity Prediction:** Integrates learned representations for
    genomic variant classification.
6.  **Explainability Analysis:** Employs Integrated Gradients,
    perturbation analysis, and attention rollout to identify influential
    genomic features.
7.  **Explanation-Guided Optimization:** Uses explanation-related
    objectives to improve the reliability and consistency of learned
    representations and predictions.
8.  **Performance Assessment:** Evaluates predictive performance and
    explanation quality through quantitative metrics and statistical
    analysis.

## 4. Project Structure

``` text
VariantExplainAI/
│
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
│
├── scripts/
│   ├── comparator_scores.py
│   ├── evaluate.py
│   ├── make_synthetic.py
│   ├── plot_results.py
│   ├── prepare_data.py
│   ├── run_ablations.py
│   ├── run_baselines.py
│   ├── run_explainability.py
│   └── train.py
│
├── src/
│   └── variantexplainai/
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── stats.py
│       │   └── temporal.py
│       │
│       ├── explainability/
│       │   ├── __init__.py
│       │   ├── assessment.py
│       │   ├── ig.py
│       │   ├── perturb.py
│       │   └── rollout.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── network.py
│       │   └── sparse_attention.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       │
│       └── utils/
│           └── __init__.py
│
├── tests/
│   └── test_pipeline.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 5. Dataset

VariantExplainAI uses genomic variant information from the **ClinVar
database**, which provides information about relationships between
genetic variations and human health conditions.

**Dataset source:** https://www.ncbi.nlm.nih.gov/clinvar/

The dataset supports the preparation of genomic variant records,
including reference alleles, alternate alleles, sequence information,
and pathogenicity-related classification labels.

The `prepare_data.py` script is responsible for preparing input data for
model development.

> **Note:** Download ClinVar data from the official source. Configure
> dataset paths, preprocessing parameters, and classification mappings
> according to the selected dataset version and experimental setup.

## 6. Installation and Environment Setup

### Prerequisites

-   Python 3.10 or later
-   pip package manager
-   Git
-   CPU or CUDA-compatible GPU (recommended for accelerated training)

### Step 1: Clone the Repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd VariantExplainAI
```

### Step 2: Create a Virtual Environment

``` bash
python -m venv venv
```

Activate the environment.

**Windows:**

``` bash
venv\Scripts\activate
```

**Linux / macOS:**

``` bash
source venv/bin/activate
```

### Step 3: Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install the Project in Development Mode

``` bash
pip install -e .
```

## 7. Configuration

The framework provides two configuration files:

  -----------------------------------------------------------------------
  Configuration                       Purpose
  ----------------------------------- -----------------------------------
  `configs/default.yaml`              Main experimental configuration

  `configs/smoke.yaml`                Lightweight configuration for
                                      pipeline validation
  -----------------------------------------------------------------------

The configuration files manage experimental settings used by the
training and evaluation pipeline.

Verify dataset locations, model parameters, training settings, and
output directories before executing experiments.

## 8. Execution Pipeline

The following scripts support the experimental workflow.

### 8.1 Data Preparation

``` bash
python scripts/prepare_data.py
```

Prepares genomic variant records for model training and evaluation.

### 8.2 Synthetic Data Generation

``` bash
python scripts/make_synthetic.py
```

Generates synthetic data for controlled pipeline testing and
development.

### 8.3 Model Training

``` bash
python scripts/train.py
```

Trains the VariantExplainAI deep learning framework.

### 8.4 Model Evaluation

``` bash
python scripts/evaluate.py
```

Evaluates the trained model using the implemented evaluation metrics.

### 8.5 Baseline Comparison

``` bash
python scripts/run_baselines.py
```

Executes baseline experiments for comparative performance assessment.

### 8.6 Ablation Studies

``` bash
python scripts/run_ablations.py
```

Evaluates the contribution of individual model components through
ablation experiments.

### 8.7 Explainability Analysis

``` bash
python scripts/run_explainability.py
```

Performs model interpretation using the implemented explainability
techniques.

### 8.8 Comparative Explanation Assessment

``` bash
python scripts/comparator_scores.py
```

Computes comparative explanation assessment scores.

### 8.9 Result Visualization

``` bash
python scripts/plot_results.py
```

Generates visual representations of experimental results.

> **Execution note:** These commands represent the intended script
> workflow. Required command-line arguments and configuration options
> should follow each script's implementation.

## 9. Explainability Methods

VariantExplainAI incorporates multiple explainability techniques to
investigate the contribution of genomic sequence features to
pathogenicity predictions.

  -----------------------------------------------------------------------
  Method                  Implementation          Purpose
  ----------------------- ----------------------- -----------------------
  Integrated Gradients    `ig.py`                 Identifies influential
                                                  input features

  Perturbation Analysis   `perturb.py`            Measures prediction
                                                  sensitivity to input
                                                  modifications

  Attention Rollout       `rollout.py`            Analyzes contextual
                                                  attention contributions

  Explanation Assessment  `assessment.py`         Evaluates explanation
                                                  quality and reliability
  -----------------------------------------------------------------------

These methods provide complementary perspectives on model interpretation
and support the analysis of prediction-related genomic features.

## 10. Evaluation

The evaluation module provides infrastructure for assessing predictive
performance and statistical reliability.

Evaluation components include:

-   Predictive classification metrics.
-   Statistical significance assessment.
-   Temporal evaluation.
-   Comparative explanation scores.
-   Explanation quality assessment.

The corresponding implementations are available in:

``` text
src/variantexplainai/evaluation/
src/variantexplainai/explainability/
```

Actual reported metric values should be generated from experimental
evaluation outputs.

## 11. Testing

The project includes a pipeline test suite. Run the tests using:

``` bash
pytest tests/
```

The test suite is intended to validate the integrated pipeline and its
major components.

For lightweight initial validation, use the configuration:

``` text
configs/smoke.yaml
```

## 12. Technologies

-   Python
-   PyTorch / Deep Learning
-   Convolutional Neural Networks
-   Transformer Architectures
-   Sparse Attention Mechanisms
-   Integrated Gradients
-   Genomic Sequence Processing
-   ClinVar Database
-   Statistical Evaluation
-   Data Visualization

## 13. Reproducibility

VariantExplainAI is organized into modular components to support
reproducible experimentation and independent evaluation.

The configuration files define experimental settings, while dedicated
scripts manage data preparation, training, evaluation, baseline
comparison, ablation studies, and explainability analysis.

For reproducibility, maintain consistent dataset versions, preprocessing
settings, model configurations, random seeds, and evaluation protocols
across experiments.




