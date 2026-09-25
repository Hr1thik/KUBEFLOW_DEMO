from kfp import dsl
from kfp.dsl import Dataset, Output

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def load_data(dataset: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(dataset.path, index=False)