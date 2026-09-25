from kfp import compiler
from pipeline import iris_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="pipeline_package.yaml"
    )
    print("Pipeline compiled successfully to pipeline_package.yaml")