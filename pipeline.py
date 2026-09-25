from kfp import dsl
from components.load_data import load_data
from components.train_model import train_model
from components.evaluate_model import evaluate_model

@dsl.pipeline(
    name="iris-training-pipeline",
    description="End-to-end Iris classification workflow"
)
def iris_pipeline():
    # Step 1: Load Data
    data_task = load_data()
    data_task.set_caching_options(False)

    # Step 2: Train Model using output dataset from Step 1
    train_task = train_model(
        dataset=data_task.outputs["dataset"]
    )
    train_task.set_caching_options(False)

    # Step 3: Evaluate Model using dataset and model artifacts
    evaluate_task = evaluate_model(
        dataset=data_task.outputs["dataset"],
        model=train_task.outputs["model"]
    )
    evaluate_task.set_caching_options(False)