from kfp.client import Client

KUBEFLOW_HOST = "http://localhost:3000"
client = Client(host=KUBEFLOW_HOST)

run = client.create_run_from_pipeline_package(
    pipeline_file="pipeline_package.yaml",
    arguments={},
    experiment_name="iris-experiments",
)

print(f"Run submitted! Run ID: {run.run_id}")