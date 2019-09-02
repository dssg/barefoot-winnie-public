from kedro.pipeline import Pipeline, node
from barefoot_winnie.d04_modelling.experiments import run_tfidf_experiment
from barefoot_winnie.d04_modelling.experiments import run_w2v_experiment
from barefoot_winnie.d04_modelling.train_winnie import train_winnie

# Defining the nodes for each model type
# User can choose the node(s) to include in the pipeline 
w2v_node = node(
		func=run_w2v_experiment,
		inputs=['primary_messages', 'w2v_experiment_settings'],
		outputs='w2v_model_results',
		name='run_w2v_model'
	)

tfidf_node = node(
	func=run_tfidf_experiment,
	inputs=['primary_messages', 'tfidf_experiment_settings'],
	outputs='tfidf_model_results',
	name='run_tfidf_model'		
	)

train_winnie_node = node(
	func=train_winnie,
	inputs=['primary_messages', 'train_winnie_settings'],
	outputs=['trained_model', 'model_numeric_vectors', 'model_raw_text'],
	name='train_winnie'		
	)

# Defining the pipeline with the specific chosen nodes
model_pipeline = Pipeline([tfidf_node, w2v_node])

# training pipeline
train_pipeline = Pipeline([train_winnie_node])

