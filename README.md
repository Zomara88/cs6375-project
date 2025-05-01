# CS 6375 Final Project: Optimizing LLM Performance in Mobile UI Navigation Through Cognitive Load Management

## Project Overview
This repository contains the implementation and analysis for our research on reducing cognitive load in LLM-based mobile UI navigation through structural pruning. We propose a lightweight preprocessing framework that identifies and removes redundant UI elements while preserving semantic structure.

## Key Features
- **Cognitive Load Analysis**: Measures UI complexity through density metrics and spatial clustering
- **Automated Pruning**: Removes noisy UI elements using optimized DBSCAN thresholds
- **Semantic Preservation**: Maintains 64% of original UI class diversity (Jaccard similarity)
- **Model-Agnostic**: Works with any LLM without retraining or task-specific tuning

## Results Highlights
- 34.3% average reduction in UI elements
- 53.3% of screens showed meaningful pruning
- Statistically significant simplification (p < 0.001) with moderate effect size (Cohen's d = 0.30)

## Environment Setup

### Kaggle:
1. Add datasets as notebook inputs:
   - `rico-dataset`
   - `annotated-rico-dataset`
   - `all-minilm-l6-v2`

### Local:
1. Download models:
   ```bash
   python -m sentence_transformers download all-MiniLM-L6-v2
