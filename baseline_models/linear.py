from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

def train_linear_model(X_train, y_train, class_weight=None):
    """Train a logistic regression model and return the pipeline."""
    model = LogisticRegression(
        solver='liblinear',
        class_weight=class_weight,
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_proba_linear(pipeline, X_test):
    """Return probability predictions from trained logistic regression pipeline."""
    return pipeline.predict_proba(X_test)[:, 1]
