"""Train a RandomForest classifier on SMILES data."""

import argparse
import numpy as np
import pandas as pd
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def get_fingerprint(smiles):
    """Generate ECFP4 fingerprint for SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        fingerprint = generator.GetFingerprint(molecule)
        binary_array = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fingerprint, binary_array)
        return binary_array
    return None


def main():
    parser = argparse.ArgumentParser(description="Train a RandomForest classifier on molecular SMILES.")
    parser.add_argument("--input", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--output", required=True, help="Path to save the trained model.pkl file.")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    data = pd.read_csv(args.input)
    
    activity_threshold = 7
    data['activity'] = data['pchembl_value_Mean'].apply(lambda x: 1 if x >= activity_threshold else 0)

    print("Generating ECFP4 fingerprints...")
    fingerprints = pd.DataFrame({smile: get_fingerprint(smile) for smile in data["SMILES"]}).T.reset_index().drop("index", axis=1)
    df = pd.concat([data, fingerprints], axis=1)

    print("Splitting dataset and training RandomForest classifier...")

    X = df.drop(['Activity_ID', 'SMILES', 'target_id', 'pchembl_value_Mean', 'activity'], axis=1)
    y = df['activity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        random_state=42,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=300
    )
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print("\nEvaluation metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


    with open(args.output, 'wb') as file:
        dump(model, file)
    print(f"\nTrained model successfully saved to {args.output}\n")


if __name__ == "__main__":
    main()
