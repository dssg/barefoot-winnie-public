from kedro.pipeline import Pipeline, node
from barefoot_winnie.d05_reporting.evaluate_model import evaluate_model

reporting_pipeline = Pipeline([
    node(
        func=evaluate_model,
        inputs=['w2v_model_results'],
        outputs=['w2v_recommendation_performance', 'w2v_performance_summary', 'w2v_performance_plots'],
        name='evaluate_model_w2v'
        ),
    node(
        func=evaluate_model,
        inputs=['tfidf_model_results'],
        outputs=['tfidf_recommendation_performance', 'tfidf_performance_summary', 'tfidf_performance_plots'],
        name='evaluate_model_tfidf'
        )
])
