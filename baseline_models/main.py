import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import *
from linear import train_linear_model, predict_proba_linear
from random_forest import train_rf_model, predict_proba_rf
from boosting import train_boosting_model, predict_proba_boosting


# ============================
#  FUNCIÓN: HISTOGRAMAS
# ============================
def plot_prediction_histograms(y_test, y_proba, model_name, target_name, output_dir):
    """
    Crea histogramas de la distribución de probabilidades predichas
    para cada clase (y=0 y y=1).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Separar probabilidades según la clase real
    proba_class0 = y_proba[y_test == 0]
    proba_class1 = y_proba[y_test == 1]

    plt.figure(figsize=(8, 5))
    
    plt.hist(proba_class0, bins=30, alpha=0.6, label="Clase 0", density=True)
    plt.hist(proba_class1, bins=30, alpha=0.6, label="Clase 1", density=True)

    plt.title(f"Distribución de Predicciones - {model_name} ({target_name})")
    plt.xlabel("Probabilidad Predicha")
    plt.ylabel("Densidad")
    plt.legend()

    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_{target_name}_hist.png")
    plt.savefig(output_path)
    plt.close()



# ============================
#  ENTRENAMIENTO MODELOS
# ============================
def train_models_for_endpoint(df, target_name, output_dir):
    """Train Linear, RF, and Boosting models, plot metrics, curves, and histograms."""
    
    # Prepare data
    X = df.drop(columns=[target_name, 'MOL_ID']).values
    y = df[target_name].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Train models
    linear_pipeline = train_linear_model(X_train, y_train, class_weight=class_weights)
    rf_pipeline = train_rf_model(X_train, y_train, class_weight=class_weights)
    boost_pipeline = train_boosting_model(X_train, y_train, scale_pos_weight=class_weights.get(1,1))

    # Get probabilities
    y_probs = {
        "Linear": predict_proba_linear(linear_pipeline, X_test),
        "Random Forest": predict_proba_rf(rf_pipeline, X_test),
        "Boosting": predict_proba_boosting(boost_pipeline, X_test)
    }

    # Compute predictions at 0.5 threshold
    y_preds = {name: (proba >= 0.5).astype(int) for name, proba in y_probs.items()}

    # Evaluate metrics at default threshold
    metrics = {name: evaluate_metrics(y_test, y_preds[name], y_probs[name]) for name in y_preds}

    # Plot ROC curves (joined)
    roc_dir = os.path.join(output_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)
    plot_roc_curves(y_test, y_probs, target_name, os.path.join(roc_dir, f"{target_name}_roc.png"))

    # Plot metrics vs threshold for each model
    for model_name, y_proba in y_probs.items():
        threshold_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
        os.makedirs(threshold_dir, exist_ok=True)
        plot_metrics_vs_threshold(y_test, y_proba, model_name, target_name, threshold_dir)

    # ================================
    # HISTOGRAMAS DE PREDICCIONES
    # ================================
    hist_dir = os.path.join(output_dir, "prediction_histograms")
    os.makedirs(hist_dir, exist_ok=True)

    for model_name, y_proba in y_probs.items():
        model_hist_dir = os.path.join(hist_dir, model_name.replace(" ", "_"))
        plot_prediction_histograms(y_test, y_proba, model_name, target_name, model_hist_dir)

    return metrics



# ============================
#  MAIN
# ============================
def main():
    endpoint_dir = "data/endpoint_csvs"
    output_dir = "model_results"
    os.makedirs(output_dir, exist_ok=True)

    target_cols = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
        'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    all_metrics = {}

    for target in target_cols:
        csv_path = os.path.join(endpoint_dir, f"{target.lower()}_endpoint.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}, skipping {target}.")
            continue

        df = pd.read_csv(csv_path)
        print(f"🔹 Training models for {target}...")

        metrics = train_models_for_endpoint(df, target, output_dir)
        all_metrics[target] = metrics

    # Print summary
    for target, metric_dict in all_metrics.items():
        print(f"\n===== {target} =====")
        for model_name, m in metric_dict.items():
            print(
                f"{model_name}: "
                f"Accuracy={m['accuracy']:.2f}, "
                f"Precision={m['precision']:.2f}, "
                f"Recall={m['recall']:.2f}, "
                f"F1={m['f1']:.2f}, "
                f"AUC={m['auc']:.2f}"
            )


if __name__ == "__main__":
    main()
