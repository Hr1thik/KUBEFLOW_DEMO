from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    df = pd.read_csv(dataset.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = joblib.load(model.path)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    # Log metrics to Kubeflow UI dashboard
    metrics.log_metric("accuracy", float(acc))