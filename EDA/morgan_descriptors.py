import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================================
# 1. Load Dataset
# ===============================================
def load_dataset(path: str) -> pd.DataFrame:
    """Load the Tox22 dataset from CSV."""
    return pd.read_csv(path)


# ===============================================
# 2. Morgan Fingerprints
# ===============================================
def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to binary Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    bitstring = fp.ToBitString()
    return np.array([int(b) for b in bitstring], dtype=np.uint8)


def compute_or_load_morgan(df: pd.DataFrame, smiles_col='SMILES',
                           output_path='morgan_descriptors.csv.gz',
                           radius=2, n_bits=1024):
    """Compute or load Morgan fingerprints."""
    if os.path.exists(output_path):
        print(f"✅ Loading precomputed Morgan descriptors from {output_path}")
        morgan_df = pd.read_csv(output_path)
        valid_idx = df.dropna(subset=[smiles_col]).index[:morgan_df.shape[0]]
        df_valid = df.loc[valid_idx].reset_index(drop=True)
        return morgan_df.drop('MOL_ID', axis=1).values, df_valid

    print("⚙️ Computing Morgan fingerprints...")
    fps = df[smiles_col].apply(lambda s: smiles_to_morgan(s, radius, n_bits))
    valid_idx = fps.dropna().index
    X = np.stack(fps.dropna().values)
    df_valid = df.loc[valid_idx].reset_index(drop=True)

    morgan_df = pd.DataFrame(X)
    morgan_df.insert(0, 'MOL_ID', df_valid['MOL_ID'])
    morgan_df.to_csv(output_path, index=False, compression='gzip')

    return X, df_valid


# ===============================================
# 3. PCA computation
# ===============================================
def compute_or_load_pca(X, df_valid, n_components=500, output_path="pca_features.csv.gz"):
    """Compute or load PCA features."""
    if os.path.exists(output_path):
        print(f"✅ Loading precomputed PCA features from {output_path}")
        pca_df = pd.read_csv(output_path)
        return pca_df.drop('MOL_ID', axis=1).values, None

    print(f"⚙️ Computing PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"💡 PCA retained {explained:.2f}% of total variance")

    pca_df = pd.DataFrame(X_pca)
    pca_df.insert(0, 'MOL_ID', df_valid['MOL_ID'])
    pca_df.to_csv(output_path, index=False, compression='gzip')

    return X_pca, pca


# ===============================================
# 4. LDA computation (1D for binary classes)
# ===============================================
def compute_all_lda(X_pca, df, target_cols, output_path="lda_features.csv.gz"):
    """
    Compute 1D LDA projection for each binary target and save all to one file.
    """
    print("⚙️ Computing LDA for all toxicity endpoints...")
    lda_results = pd.DataFrame({'MOL_ID': df['MOL_ID'].values})
    lda_models = {}

    for target in target_cols:
        y = df[target].values
        valid_idx = ~pd.isna(y)
        X_valid = X_pca[valid_idx]
        y_valid = y[valid_idx]

        # Fit 1D LDA (binary -> 1 component)
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = np.full(len(df), np.nan)
        X_lda[valid_idx] = lda.fit_transform(X_valid, y_valid).ravel()

        lda_results[target] = X_lda
        lda_models[target] = lda
        print(f"✅ LDA computed for {target}")

    lda_results.to_csv(output_path, index=False, compression='gzip')
    print(f"💾 All LDA features saved to {output_path}")
    return lda_results, lda_models


# ===============================================
# 5. 1D LDA visualization
# ===============================================
def lda_visualization_1d(lda_results: pd.DataFrame, df: pd.DataFrame,
                         target_cols: list, output_dir="morgan_plots/lda"):
    """
    Plot 1D LDA projection as KDE plots for each binary endpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    for target in target_cols:
        if target not in lda_results.columns:
            continue
        valid_idx = ~pd.isna(lda_results[target])
        if valid_idx.sum() == 0:
            continue

        df_plot = pd.DataFrame({
            'LD1': lda_results.loc[valid_idx, target],
            target: df.loc[valid_idx, target]
        })

        plt.figure(figsize=(7,5))
        sns.kdeplot(
            data=df_plot, x='LD1', hue=target,
            fill=True, common_norm=False, alpha=0.6, palette='coolwarm'
        )
        plt.title(f"LDA 1D Separation - {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"LDA_{target}_1D.png"), dpi=300)
        plt.close()
        print(f"✅ Saved 1D LDA density plot for {target}")


# ===============================================
# 6. Main
# ===============================================
if __name__ == "__main__":
    path = "data/tox22.csv"
    df = load_dataset(path)

    target_cols = [
        'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
        'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
    ]

    X, df_valid = compute_or_load_morgan(df, smiles_col='SMILES', output_path='morgan_descriptors.csv.gz')
    X_pca, pca_model = compute_or_load_pca(X, df_valid, n_components=500, output_path='pca_features.csv.gz')

    lda_results, lda_models = compute_all_lda(X_pca, df_valid, target_cols, output_path='lda_features.csv.gz')
    lda_visualization_1d(lda_results, df_valid, target_cols, output_dir="morgan_plots/lda")

