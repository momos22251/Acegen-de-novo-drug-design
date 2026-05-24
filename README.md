# Acegen ML - Nextflow Pipeline

A standalone Nextflow (DSL2) pipeline for target-conditioned de-novo drug design. It orchestrates training a classifier, generating molecules via reinforcement learning (using REINVENT/ACEGEN), and plotting results.

---

## Environment Setup

Follow these steps to set up your environment:

1. **Install Conda**: Ensure Conda is installed on your system. If not, follow the installation instructions on [Miniconda](https://docs.anaconda.com/miniconda/install/).
2. **Create the Environment**: Create the `acegen` environment using the provided `environment.yaml` file:
   ```bash
   conda env create -f environment.yaml
   ```
3. **Activate the Environment**:
   ```bash
   conda activate acegen
   ```
4. **Install ACEGEN**: Install ACEGEN from the official repository without dependencies:
   ```bash
   pip install git+https://github.com/Acellera/acegen-open.git --no-deps
   ```

---

## Running the Pipeline

Ensure [Nextflow](https://www.nextflow.io/) (version `22.10.0` or higher) is installed, then run:

```bash
nextflow run main.nf
```

---

## Project Structure

*   `main.nf`: Nextflow workflow orchestrator.
*   `nextflow.config`: Pipeline configurations (environment, error handling, inputs).
*   `bin/`: Executable helper scripts (Random Forest training, result analysis/plotting).
*   `configs/`: Configuration files for reinforcement learning optimization.
*   `data/`: Training datasets, target details, and intermediate classifier model.
*   `scripts/`: Reinforcement learning script (`reinvent.py`).

---

## Outputs

Upon successful completion, the pipeline generates:

*   `data/model.pkl`: The trained classifier.
*   `results/`:
    *   `RFC_reinvent_<timestamp>/`: RL logs, TensorBoard events, and model checkpoints.
    *   `rfc_average_score_steps.png`: Average score progress plot.
    *   `rfc_top_molecules.png`: 2D drawings of top-scoring active molecules.
