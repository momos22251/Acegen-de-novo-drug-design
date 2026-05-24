# Acegen ML - Nextflow Pipeline

This directory contains a standalone Nextflow (DSL2) pipeline for target-conditioned de-novo drug design. It orchestrates the process of training a classifier, generating molecules via reinforcement learning optimization (using REINVENT/ACEGEN), and plotting results.

## Prerequisites

Before running the pipeline, ensure you have the following installed and configured:

1. **Nextflow**: Version `22.10.0` or higher.
2. **Conda**: A Conda environment named `acegen` containing the necessary dependencies (`rdkit`, `torchrl`, `scikit-learn`, `molscore`, etc.). The default environment location is assumed to be `/home/mohamed/miniconda3/envs/acegen` (as configured in `nextflow.config`).

## Project Structure

*   `main.nf`: The main Nextflow workflow orchestrator.
*   `nextflow.config`: Nextflow configuration specifying the environment, error strategy, and default input paths.
*   `bin/`: Contains executable helper scripts (automatically added to the `$PATH` by Nextflow):
    *   `train_classifier.py`: Extracts molecular ECFP4 fingerprints and trains a Random Forest Classifier.
    *   `analyze_results.py`: Computes average reinforcement learning score progress, plots performance, and generates 2D drawings of top active molecules.
*   `configs/`: Configuration files for reinforcement learning optimization (including `RFC_config_denovo.yaml` and `RFC.json`).
*   `data/`: Directory containing training datasets, protein targets, and ligands.
*   `scripts/`: Contains the reinforcement learning script (`reinvent.py`).

## Running the Pipeline

To run the pipeline from this directory, execute:

```bash
nextflow run main.nf
```

### Detached Background Execution
If running on a remote server or in the background, you can use `setsid` or redirect standard descriptors to prevent terminal signals from suspending the process:

```bash
setsid nextflow run main.nf > nextflow_run.log 2>&1
```

## Outputs

When the pipeline completes successfully, the following outputs will be created:

*   `data/model.pkl`: The trained Random Forest Classifier model.
*   `results/`:
    *   `RFC_reinvent_<timestamp>/`: The generated scores CSV, tensorboard records, and model checkpoints from reinforcement learning.
    *   `rfc_average_score_steps.png`: Line plot tracking average score progress over RL training steps.
    *   `rfc_top_molecules.png`: Grid image showing the 2D structures of the top 3 generated active molecules.
