# Udacity_Disaster_response

## Introduction

This repo is for the Disaster reponse project of the Udacity data science nanodegree.

## Dependencies:
- Python 3.5+
- NLTK for natural language processing (converting text data into numerical data)
- Pandas, Numpy, scikit-learn, sqlalchemy for data processing and machine learning
- Matplotlib, seaborn, plotly for data visualizations
- Flask, back-end of our minimalistic web app

## Instructions:

Run the following commands in the project's root directory to set up the database and model.

1. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl Note by default the training data will be up-sampled before training. You can change this setting in the train_classifier.py by setting ML_classifier(df, sample = False) when instantiating the ML model class. It should take less than a minute to train and save the model.

3. un the following command in the app's directory to run the web app python run.py

4. Go to http://0.0.0.0:3001/ to use the web app to query your own message and see some visualizations about the original dataset.
