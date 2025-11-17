from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_rf_model(X_train, y_train, class_weight=None):
    """Train a Random Forest model and return the pipeline."""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_proba_rf(pipeline, X_test):
    """Return probability predictions from trained Random Forest pipeline."""
    return pipeline.predict_proba(X_test)[:, 1]
