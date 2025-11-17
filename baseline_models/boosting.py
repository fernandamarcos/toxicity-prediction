from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_boosting_model(X_train, y_train, scale_pos_weight=4):
    """Train a LightGBM boosting model and return the pipeline."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_proba_boosting(pipeline, X_test):
    """Return probability predictions from trained LightGBM pipeline."""
    return pipeline.predict_proba(X_test)[:, 1]
