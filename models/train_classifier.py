import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
import pickle


def load_data(database_filepath):
    """load data from database
    INPUT: file path of the database    
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
    """Tokenize text
    """
    # define regular expression for url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    

# define a scoring function for grid search
def mean_accuracy(y_true,y_pred):
    accuracy = [] # list to store accuracy of each column
    y_pred = pd.DataFrame(y_pred,columns = y_true.columns)
    for col in y_true.columns:
        accuracy.extend([np.mean(y_true[col].values==y_pred[col].values)])
    return np.mean(accuracy)

def build_model():
    # build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # define a scorer for the multioutput classification
    # scorer = make_scorer(mean_accuracy,greater_is_better = True)

    # set up grid search to find the best model
    # parameters for grid search
    parameters = {
    "clf__estimator__max_depth":(50,100,200),
    "clf__estimator__bootstrap":(True,False),
    "vect__stop_words": ('english',None)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 2, n_jobs = -1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performace of the final model
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred,columns = category_names)
    for col in category_names:
        print('Classifer report of {}\n'.format(col))
        print(classification_report(Y_test[col], y_pred_df[col]))
    return

def save_model(model, model_filepath):
    """Save the final model using pickle
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    return

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
