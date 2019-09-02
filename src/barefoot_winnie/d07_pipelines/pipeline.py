from kedro.pipeline import Pipeline
from barefoot_winnie.d07_pipelines.intermediate import int_pipeline
from barefoot_winnie.d07_pipelines.primary import prm_pipeline
from barefoot_winnie.d07_pipelines.model import model_pipeline
from barefoot_winnie.d07_pipelines.model import train_pipeline
from barefoot_winnie.d07_pipelines.reporting import reporting_pipeline


def create_train_pipeline():
    """Create the pipeline for training winnie.

    Returns:
        Pipeline: The resulting pipeline.
    """
    pipeline = Pipeline([
            int_pipeline,
            prm_pipeline,
            train_pipeline,
    ])

    return pipeline


def create_pipeline():
    """Create the model selection pipeline.

    Returns:
        Pipeline: The resulting pipeline.

    """
    pipeline = Pipeline([
        int_pipeline,
        prm_pipeline,
        model_pipeline,
        reporting_pipeline
    ])
    
    return pipeline



