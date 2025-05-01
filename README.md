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

## References

1. **RICO Dataset**  
   ```bibtex
   @article{deka2017rico,
     author    = {Deka, Biplab and Huang, Zifeng and Franzen, Carl and Hibschman, John and Li, Yang and Nichols, Jeffrey and Kumar, Ranjitha},
     title     = {Rico: A Mobile App Dataset for Building Data-driven Design Applications},
     booktitle = {Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology (UIST)},
     year      = {2017},
     pages     = {845--854},
     publisher = {ACM},
     doi       = {10.1145/3126594.3126651}
   }

2. **Annotated RICO Dataset**
   ```bibtex
   @article{sunkara2022,
     author    = {Srinivas Sunkara and Maria Wang and Lijuan Liu and Gilles Baechler and Yu-Chung Hsiao and Jindong Chen and Abhanshu Sharma and James Stout},
     title     = {Towards Better Semantic Understanding of Mobile Interfaces},
     journal   = {CoRR},
     volume    = {abs/2210.02663},
     year      = {2022},
     url       = {https://arxiv.org/abs/2210.02663}
   }
