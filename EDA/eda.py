"""
This file processes the tox22 dataset for toxicology analysis.
It includes functions for loading the dataset, handling missing values,
and performing exploratory data analysis (EDA) such as visualizations and
statistical summaries. Figures are saved instead of shown.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Tox22 dataset from a CSV file and add molecule IDs.
    """
    df = pd.read_csv(path)
    
    # Drop any completely empty rows at the end
    df.dropna(how='all', inplace=True)
    
    # Add molecule IDs
    if "MOL_ID" not in df.columns: 
        df.insert(0, 'MOL_ID', [f'MOL_{i:05d}' for i in range(len(df))])
    
    # Save the modified dataset
    df.to_csv(path, index=False)
    return df

def dataset_summary(df: pd.DataFrame):
    """Print basic information about the dataset."""
    df_analysis = df.drop('MOL_ID', axis=1)
    print("Shape:", df_analysis.shape)
    print("\nData types:")
    print(df_analysis.dtypes)
    print("\nMissing values:")
    print(df_analysis.isna().sum())

def class_balance(df: pd.DataFrame, endpoints: list):
    """Save the class balance plots for each toxicity endpoint."""
    endpoints = [col for col in endpoints if col != 'MOL_ID']

    for col in endpoints:
        counts = df[col].value_counts(dropna=False)
        plt.figure()
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title(f"Class distribution for {col}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.savefig(f"plots/class_balance_{col}.png", bbox_inches="tight")
        plt.close()

def compute_descriptors(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """
    Compute basic molecular descriptors from SMILES strings.
    Adds columns to the DataFrame: MolWt, LogP, NumHDonors, NumHAcceptors.
    """
    mol_wt, logp, h_donors, h_acceptors = [], [], [], []

    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol_wt.append(Descriptors.MolWt(mol))
            logp.append(Descriptors.MolLogP(mol))
            h_donors.append(Descriptors.NumHDonors(mol))
            h_acceptors.append(Descriptors.NumHAcceptors(mol))
        else:
            mol_wt.append(np.nan)
            logp.append(np.nan)
            h_donors.append(np.nan)
            h_acceptors.append(np.nan)

    df["MolWt"] = mol_wt
    df["LogP"] = logp
    df["HDonors"] = h_donors
    df["HAcceptors"] = h_acceptors
    return df

def plot_descriptors(df: pd.DataFrame, descriptors: list):
    """Save histograms for selected molecular descriptors."""
    for desc in descriptors:
        plt.figure()
        sns.histplot(df[desc].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {desc}")
        plt.xlabel(desc)
        plt.ylabel("Frequency")
        plt.savefig(f"plots/distribution_{desc}.png", bbox_inches="tight")
        plt.close()

def correlation_analysis(df: pd.DataFrame, endpoints: list):
    """Save correlation heatmap for toxicity endpoints."""
    corr = df[endpoints].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between endpoints")
    plt.savefig("plots/correlation_endpoints.png", bbox_inches="tight")
    plt.close()

def dimensionality_reduction(df: pd.DataFrame, features: list, label: str):
    """
    Apply PCA and t-SNE to visualize chemical space.
    Colors points by the given label. Saves the plots.
    """
    X = df[features].dropna()
    y = df.loc[X.index, label]

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", alpha=0.7)
    plt.title(f"PCA projection colored by {label}")
    plt.savefig(f"plots/pca_{label}.png", bbox_inches="tight")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette="Set1", alpha=0.7)
    plt.title(f"t-SNE projection colored by {label}")
    plt.savefig(f"plots/tsne_{label}.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Load dataset
    df = load_dataset("./data/tox22.csv")

    # Identify toxicity endpoints (all binary labels except SMILES)
    endpoints = [col for col in df.columns if col != "SMILES" and col !="MOL_ID"]

    # Summary
    dataset_summary(df)

    # Class balance
    class_balance(df, endpoints)

    # Compute descriptors
    df = compute_descriptors(df, "SMILES")

    # Plot descriptor distributions
    plot_descriptors(df, ["MolWt", "LogP", "HDonors", "HAcceptors"])

    # Correlation analysis
    correlation_analysis(df, endpoints)

    # Dimensionality reduction
    dimensionality_reduction(df, ["MolWt", "LogP", "HDonors", "HAcceptors"], endpoints[0])