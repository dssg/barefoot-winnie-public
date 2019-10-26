# Response Recommendation System for BarefootLaw

The aim of this project is to improve the first-line legal aid processes of BarefootLaw. 

## Table of Contents

1. [Introduction](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#introduction)
2. [System overview](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#system_overview)
2. [Installation and setup](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#installation)
3. [Code organization](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#code_organization)
4. [Training pipeline](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#training_pipeline)
5. [Inference pipeline](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#inference_pipeline)
6. [Testing](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#testing)
7. [Contributors](https://github.com/dssg/barefoot_winnie/blob/dev/README.md#contributors)


## Introduction

### Data Science for Social Good (DSSG) at Imperial College London

The Data Science for Social Good Summer Fellowship is a summer program organized by the Data Science for Social Good Foundation and partnering organizations to train aspiring data scientists to work on data mining, machine learning, big data, and data science projects with social impact. Working closely with governments and nonprofits, fellows take on real-world problems in education, health, energy, public safety, transportation, economic development, international development, and more.

## System Overview

In Uganda, geographical and financial barriers limit people’s access to legal advice and guidance. However, mobile phone technology is widespread in the country. BarefootLaw (BFL) is a not for profit organization that leverages the abundance of mobile phones to provide free legal guidance to Ugandans via social media and SMS. People have been quick to make use of BFL’s services and as a result, the number of requests has been growing every year. Furthermore, BFL has an ambition of expanding and reaching 50 million people by 2030. BFL is a relatively small team and the increasing workload has led to an average response time of 72 hours per question. In this project, we propose an Artificial Intelligence (AI) system that can help make the response drafting process more efficient and potentially help BFL cut down their response time to 24 hours. More specifically, we developed a system that takes an incoming question, and provides a set of candidate responses that the lawyer could use/edit to draft the response to a beneficiary. The system is called Winnie, and it was approached as an information retrieval system where the question is the query and the historical question-answer pairs are the documents to be retrieved. The system was based on the text data of historical question-answer pairs provided by BFL.

BarefootLaw receives legal questions from people through three different written channels: 1) Facebook, 2) SMS, and 3) email. The developed system is intended to speed up the process of processing a question and generating a answer to be sent back to the beneficiary. The system will estimate a set of candidate answers to a question that a lawyer can edit and send back to the beneficiary.

The incoming requests are stored in a MySQL database. The developed system---Winnie---takes the incoming question, preprocesses the text, and converts the natural language text to a structured representation. Then, structured representations of questions are fed into a machine learning model to estimate the answer to a given question. These estimated answers are written back to the MySQL database and displayed to the lawyer through a webpage. 

## Installation and setup

1. Install Anaconda:
The codebase is built on Anaconda environment. The following commands are to be used for installing:

```
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
$ bash Anaconda3-2019.03-Linux-x86_64.sh
$ source ~/.bashrc
```

2. Install Python packages:
The system is developed in Python and the code is dependent on several Python packages. All the Python Packages needed for the system are listed in the `requirements.txt` file. In order to run the pipeline, it is advisable to create a new python environment with Python version 3.7. The virtual environment can be created using any virtual enviroment manager. If using Anaconda, the following commands can be used to create and activate the environment. 

``` 
$ conda create --name winnie python=3.7
$ conda activate winnie 
```

In order to clone and access the repository, the following commands should be run.

```
$ git clone https://github.com/dssg/barefoot_winnie.git
$ cd barefoot_winnie
```

Then, once the environment is created and activated, the following command can be used to install the packages

`$ pip install -r requirements.txt`

3. Install the NLTK resources
In addition to the python packages listed in requirements.txt, the system is dependent on the availability of several Natural Lanuage Tool Kit (NLTK) resources. setup.py script can be used to install the required resources by running the following command from the command line. 

` $ python setup.py `

4. Set up credentials to access the MySQL database
This codebase is built to run on a specific dataset. Anyone who wants to run the codebase on the original dataset has to request access to BarefootLaw. Once the git repository is cloned and in order to access the dataset, the system reads a credentials file that should be in a specific location. The following commands access this folder.

`$ cd barefoot_winnie/conf/local/`

A _credentials.yml_ file should be created using the following format:

```
dssg:
 con: connection_to_MySQL_database. Format (no quotes): 'mysql+pymysql://' + BFL_username + ':' + BFL_password + '@' + host + '/bios'
 
 username: BFL_username
 password: BFL_password
```

Note that there is single space identation in some of the fields. A _.yml_ file can be created from the notepad by changing the ending format. The credentials file should be set up before the following pipelines are run.

## Training pipeline

Training pipeline trains the natural language models and machine learning models and saves them to the disk. The methods in this pipeline pull all the existing messages in MySQL database and create a model for Winnie to look into when a new inquiry comes in.

The first part of the training pipeline imports the following tables from MySQL:
  - messages
  - sent_sms
  - received_sms
  
Example of format of the tables can be found in the following URL: https://github.com/dssg/barefoot_winnie/wiki/III.-Data-Preparation.

The rest of the training pipeline consists in: i) cleansing, formatting and combining, ii) preprocessing steps, iii) feature generation with TFIDF/W2V, iv) save the train model and the text related to the model.

After installation and setup, we are ready to train our model. The training pipeline is setup with Kedro (Kedro is installed as a part of the requirements). Documentation on kedro pipeline: https://kedro.readthedocs.io/en/latest/.  The following command can be used to run the pipeline - user should go to _barefoot_winnie/_ folder.

`$ kedro run`

Alternatively, the training pipeline can be set run automatically every day at 2am using the following command:
```
python src/cron_job.py
```

The parameters of training can be modified by editing the `conf/base/parameters.yml` file and the *train_winnie_settings* part. The parameters are described below:
  - *feature_type:* TFIDF or Word2Vec
  - *preprocessing_steps:* which preprocessing steps to include on the raw text. For detailed information on preprocessing steps refer to: https://github.com/dssg/barefoot_winnie/wiki/IV.-Feature-Generation.
  - *train_params:* max_df, min_df for TFIDF
  - *w2v_pretrained_file:* When running Word2Vec, this is the path to store the download pretrained vector. The Word2Vec vector can be downloaded in the following URL: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz.

The storage path of the trained models can be modified by editing the `conf/base/catalog.yml` file in the *Trained Models* part. Three files are saved:
  - trained_model: pickle file. Saved features and hyperparameters for selected model (default: TFIDF).
  - model_numeric_vectors: parquet file. Numerical representation of the training questions.
  - model_raw_text: parquet file. Text representation of training questions and answers.

## Inference pipeline

The inference pipeline executes when the Lawyers need the estimations from Winnie to a new question. The inference pipeline brings the case id -integer key to identify case- of a specific question directly from MySQL database and outputs the recommended responses. In specific, the column *id* from the **cases** table from MySQL.

The following should be run in the command line in order to run the inference pipeline:

```
$ python
>>> import os
>>> os.chdir(os.path.join(os.getcwd(), 'src'))
>>> from barefoot_winnie.d05_reporting.run_winnie import run_winnie
>>> run_winnie(case_id=4796)
```

The user should replace **4796** by the new case id they want recommendations for.

The 5 candidate responses to be displayed will be written in the table named `recommended_responses_3` in the MySQL database. This database contains the following columns:
  - case_id: id of incoming case
  - recommended_response: candidate response
  - response_rank: ranking (**int.** 1 to 5) of the recommended response, where 1 is the most recommended and 5 the least one

Each case_id will have five recommended responses. Therefore, each case_id will be repeated five times in that table.

## Testing
This codebase is tested on Linux Ubuntu 18.04.2 LTS environment. If the codebase is to be run on new data with the same structure as the BIOS database, the user should only change the credentials file to have access to the new mysql database. If the tables have different format and names, the _barefoot_winnie/conf/base/catalog.yml_ should be changed in the _RAW_ section to receive the names of new tables in the 'table_name' field.

Perfomance metrics were estimated for different settings on the input data. The results can be seen below:

![score](https://github.com/dssg/barefoot_winnie/blob/dev/images/Evaluation_score.png)

## Code Organization

The code is organized into 8 submodules. 

1. d00_utils: Utility functions used throughout the system
2. d01_data: Importing data from the MySQL dataset
3. d02_intermediate: Pre-processing the raw data
4. d03_primary: Creating the dataset for buildng the model from the intermediate data
5. d04_modelling: Feature engineering, training/testing of the models
6. d05_reporting: Producing a set of metrics for validating the model's results
7. d06_visualisation: Generating plots for result reporting 
8. d07_pipelines: defining the pipeline nodes




## Contributors

Data Science for Social Good 2019 Fellows:
Kasun Amarasinghe, 
Carlos Caro, 
Nupoor Gandhi, 
Raphaelle Roffo

Technical Mentor: Maren Eckhoff

Project Manager: Samantha Short
