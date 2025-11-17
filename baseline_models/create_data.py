import os
import pandas as pd

def load_data():
    """Load PCA features and the original dataset."""
    pca_features = pd.read_csv("pca_features.csv.gz")
    original_data = pd.read_csv("data/tox22.csv")
    return pca_features, original_data

def create_endpoint_csvs(pca_features, original_data, output_dir="data/endpoint_csvs"):
    """Create separate CSV files for each endpoint."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define target columns
    target_cols = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
        'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    for target in target_cols:
        print(f"Processing {target}...")
        
        # Merge PCA features with the original data to get the target variable
        dataset = pd.merge(pca_features, original_data[['MOL_ID', target]], on='MOL_ID', how='inner')
        
        # Drop rows where the target variable is missing
        dataset.dropna(subset=[target], inplace=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{target.lower()}_endpoint.csv")
        dataset.to_csv(output_path, index=False)
        
        # Print info about the saved dataset
        print(f"✅ Saved {output_path}")
        print(f"   Samples: {len(dataset)}")
        print(f"   Class distribution: \n{dataset[target].value_counts()}\n")

def main():
    # Load data
    print("Loading data...")
    pca_features, original_data = load_data()
    
    # Create endpoint-specific CSVs
    print("\nCreating endpoint-specific CSVs...")
    create_endpoint_csvs(pca_features, original_data)
    
    print("💾 All endpoint CSVs created successfully!")

if __name__ == "__main__":
    main()