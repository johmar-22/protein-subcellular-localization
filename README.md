# Attention-Enhanced Bi-LSTM Ensembles with Frozen ESM-2 Embeddings for Protein Subcellular Localization

**Author:** Johaimen M. Omar

## Overview
This repository contains the complete code, data splits, and results for the
study presented in the manuscript. The model combines frozen ESM-2 protein
language model embeddings with a Bidirectional LSTM + attention classifier
to predict protein subcellular localization into four classes:
Nucleus, Cytoplasm, Mitochondrion, and Cell Membrane.

## Repository Structure
```
protein-subcellular-localization/   ← name this after your paper
│
├── README.md                         ← see Section 5 for what to write
├── requirements.txt                  ← see Section 6
├── main.py                           ← renamed from the .py file
├── main_notebook.ipynb               ← renamed from the .ipynb file
│
├── data/
│   ├── train_split.csv               ← Supplementary Table S1 (training)
│   ├── val_split.csv                 ← Supplementary Table S1 (validation)
│   ├── test_split.csv                ← Supplementary Table S1 (test)
│   ├── uniprot_strict_v6.fasta       ← raw source sequences
│   ├── embedding_metadata.json       ← memmap array shapes
│   └── benchmark/
│       ├── benchmark_sequences.fasta             ← Supplementary Table S2
│       ├── scientific_benchmark_results.csv
│       ├── deeploc_2.1_results.csv
│       ├── final_muloc_comparison.csv
│       ├── muloc_subset_metadata.csv
│       └── muloc_results.txt
│
├── models/
│   ├── label_encoder.pkl
│   └── README_models.md              ← explains how to download weights
│
├── results/
│   ├── statistical_results.csv
│   ├── svm_baseline_results.json
│   └── svm_classification_report.txt
│
└── figures/
    ├── Figure1_SpeciesDistribution.pdf
    ├── Figure2_ConfusionMatrix.png
    ├── Figure3_ROC.png
    ├── Figure4_Best_Membrane.png
    ├── Figure4_Best_Mitochondrion.png
    ├── Figure4_Best_Nucleus.png
    ├── Figure4_Best_Cytoplasm.png
    └── Figure5_tSNE_Clusters.png
```

## Reproducing the Results

### 1. Install dependencies
   pip install -r requirements.txt

### 2. Regenerate ESM-2 embeddings (Part 5 of main.py)
   The numpy.memmap embedding files (X_train.dat, X_val.dat, X_test.dat)
   are not stored in this repository due to their size (~15 GB combined).
   They can be regenerated from the provided data splits using:
     python main.py  # requires GPU; runs Part 5 (ESM-2 Extraction)
   The ESM-2 model (facebook/esm2_t33_650M_UR50D) will be downloaded
   automatically from HuggingFace.

### 3. Download pre-trained model weights
   Download from GitHub Releases:
     https://github.com/YOUR_USERNAME/repo/releases/download/v1.0.0/fold_checkpoints.zip
   Unzip into: fold_checkpoints/

### 4. Run evaluation
   python main.py  # runs Parts 8–10 (evaluation, benchmark, stats)

## Data
- data/train_split.csv / val_split.csv / test_split.csv — Supplementary Table S1
  (UniProt accession IDs, sequences, class labels, CD-HIT cluster IDs)
- data/benchmark/benchmark_sequences.fasta — Supplementary Table S2
  (86 prospective eukaryotic proteins released in UniProtKB after Jan 1, 2024)
- data/uniprot_strict_v6.fasta — raw source sequences downloaded from UniProt
  using the Boolean queries in the Methods section (SL-0191, SL-0086, SL-0173, SL-0039)

## PAPER UNDER REVIEW
