# Proteomic and Genomic Drug Development

## Introduction 
This project encompasses an end-to-end computational biology pipeline focused on **Proteomic and Genomic Drug Development**. By utilizing state-of-the-art machine learning, deep learning, and transformer-based models, this suite provides tools for analyzing, predicting, and generating biological sequences (DNA and Proteins) and facilitating targeted, personalized drug development.

## Key Components & Methodologies

Our repository is divided into four main independent but complementary modules:

### 1. Protein Generation & Sequencing
We provide two distinct approaches for custom protein sequence generation and analysis:
- **Approach 1 (RNN-based)**: Uses a custom Enhanced Bidirectional LSTM with an Attention mechanism for *de novo* protein sequence generation.
- **Approach 2 (Transformers-based)**: Employs `lamm-mit/ProteinForceGPT` via a Streamlit interface to generate protein sequences based on prompts, calculate structural forces, and heuristically estimate protein stability.

### 2. Protein-Protein Interaction (PPI) Analysis
An essential component for understanding molecular mechanics and drug targeting:
- Powered by a fine-tuned `facebook/esm2_t33_650M_UR50D` model.
- Includes a Gradio UI to predict interaction probabilities between sequences, identify virus-host PPIs, and detect severe interaction disruptions caused by specific amino acid mutations.

### 3. Genome Prediction & Generation
A suite dedicated to DNA sequence manipulation and deep biological analysis:
- Uses `AIRI-Institute/gena-lm-bert-base` for DNA sequence completion and detailed sequence classification (e.g., Coding vs. Non-coding, Regulatory regions).
- Uses `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` for sequence analysis (extracting activity levels, genetic variation scores, and genome accessibility).

### 4. Drug Generation for Target Proteins
Targeted towards personalized drug development by generating possible small molecules for specific protein targets:
- Uses the `alimotahharynia/DrugGen` (GPT-2 based) generative model.
- Automatically generates drug-like SMILES sequences given protein sequences or UniProt IDs.
- Calculates and evaluates drug-likeness based on Lipinski's Rule of Five using `rdkit`.
- Features a Gradio UI that provides automatic 2D visualizations and interactive 3D chemical structures via `py3Dmol`.

## Validation & Evaluation
Outputs generated within the core pipelines have conceptually and practically been structurally validated using advanced folding predictions, predominantly relying on **AlphaFold2**. Generated structures are evaluated utilizing strict qualitative metrics like the Predicted local Distance Difference Test (**pLDDT**) and predicted Template Modeling (**pTM**) to ensure confidence and correct folding behavior.

## Repository Structure

- `Genome Prediction and Generation/`: BERT and Nucleotide Transformer scripts and UIs for comprehensive DNA analysis.
- `PPI Analysis - Approach 2/`: ESM2 model scripts and interactive Gradio interface for protein interaction prediction.
- `RNN Protein Generation- Approach 1/`: Data preprocessing, the custom Bidirectional LSTM model architecture, and training scripts.
- `Reinforcement GAN for Drug Generation - Approach 2/`: GPT-2 based small molecule generation pipeline mapped to UniProt targets.
- `Sequence Generation - Approach 2/`: ProteinForceGPT inference scripts presented via Streamlit.
- `Notebooks/` & `Figures/`: Exploratory data analysis, evaluations, and visualizations mapping from our initial approaches.
- `Proteomic and Genomic Drug Development.pdf`: The core foundational document elaborating on the theoretical underpinnings of the implemented tools.

## Requirements & Setup

To run the tools and interactive web UIs across the different modules, install the essential global dependencies:

```bash
pip install torch transformers rdkit py3Dmol gradio streamlit stmol scikit-learn pandas numpy matplotlib
```

*(Note: Certain approaches may require additional packages and older dependencies for complete reproducibility. We highly recommend using a PyTorch-enabled Conda environment with dedicated CUDA support for fast model inference).*

## Acknowledgments
This project contributes to the rapidly growing intersection of Generative AI and Biotechnology. It aims to significantly reduce the computational time required in the early stages of drug discovery through *in-silico* generation, advanced sequence classification, and rapid validation.
