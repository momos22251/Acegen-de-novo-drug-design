#!/usr/bin/env python
"""Analyze Acegen optimization results."""

import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyze results and plot scores for the Acegen pipeline.")
    parser.add_argument("--results-dir", required=True, help="Path to the results directory to search for scores.csv.")
    parser.add_argument("--output-dir", required=True, help="Path to save the generated plots and visualizations.")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


    results_pattern = str(results_path / "RFC_reinvent_*/*/scores.csv")
    result_files = glob.glob(results_pattern)

    if result_files:
        latest_scores_file = sorted(result_files)[-1]
        print(f"\nLoading results from: {latest_scores_file}")
        rfc_df = pd.read_csv(latest_scores_file, index_col=0)
    else:

        fallback_path = results_path / "RFC_reinvent_2025_01_23_183310/2025_01_23_reinvent_RFC/scores.csv"
        if fallback_path.exists():
            print(f"\nLoading fallback results from: {fallback_path}")
            rfc_df = pd.read_csv(fallback_path, index_col=0)
        else:
            print(f"\nNo generation results found in {results_path}. Exiting.")
            return

    print(f"MAX predicted probability: {rfc_df['RFC_pred_proba'].max()}")
    print(f"MEDIAN predicted probability: {rfc_df['RFC_pred_proba'].median()}")
    

    active_molecules_rfc = rfc_df[
        (rfc_df['RFC_pred_proba'] >= 0.5) & (rfc_df['unique'] == True)
    ].sort_values('RFC_pred_proba', ascending=False).reset_index().drop("index", axis=1)
    
    print("\nTop generated active molecules:")
    print(active_molecules_rfc.head())


    grouped_data = rfc_df.groupby(['step'])['RFC_pred_proba'].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(grouped_data.index, grouped_data.values, marker='o', linestyle='-', color='b')
    plt.xlabel('Step')
    plt.ylabel('Mean RFC predicted probability')
    plt.title('Average Score across Training Steps')
    plt.grid(True)
    
    plot_file = output_path / "rfc_average_score_steps.png"
    plt.savefig(plot_file)
    print(f"Saved training plot to {plot_file}")
    plt.close()


    active_molecules_rfc['molecules'] = active_molecules_rfc['smiles'].apply(Chem.MolFromSmiles)
    valid_active = active_molecules_rfc.dropna(subset=['molecules'])
    molecules = valid_active['molecules'].tolist()[:3]
    legends = valid_active['smiles'].tolist()[:3]
    
    if molecules:
        molecules_image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=3,
            subImgSize=(350, 350),
            legends=legends
        )
        image_file = output_path / "rfc_top_molecules.png"
        molecules_image.save(image_file)
        print(f"Saved molecules visualization to {image_file}")
    else:
        print("No valid molecules to visualize.")


if __name__ == "__main__":
    main()
