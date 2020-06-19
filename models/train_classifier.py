import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """Load dataframe from database
    
    INPUT
    database_filepath: Link to the database
    
    OUTPUT
    X: Messages column
    y: Categories columns
    category_names: The columns for the categories
    """
        
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('project_disasters', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X,y,category_names


def tokenize(text):
    """
    Tokenize function
    
    INPUT
    tex: list of text messages (english)
    
    OUTPUT
    clean_tokens: normalized and lemmatized tokens for the ML algorithm
    """
    
    # normalize case and remove punctuation/ Removed to be able to run the starting verb extractor
    #text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class - Custom class to extract the starting verb of a sentence, creating a new feature for the ML classifier
    """
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
   
def build_model():
    
    """
    Build Model function
    
   INPUT
    None
    
   Output
   A Scikit Learn pipeline for processing text messages and returning a classifier
    """
    #Create the Pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('starting_verb', StartingVerbExtractor())
            
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #Change the parameters to improve the model
    parameters = {
    'clf__estimator__max_depth': [10, 50, 100, None],
    'clf__estimator__max_features': ['auto', 'sqrt'],
    'clf__estimator__min_samples_leaf':[2, 5, 10]
    }
    
    #Use GridSearchCV to improve the model with the parameters
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate the model using - f1 score, precision and recall
    INPUT
    model: the pipeline
    X_test: testing messsages
    y_test: testing classification
    category_names = required, list of category strings
    OUTPUT
    Prints the metrics
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save Model
    
    INPUT
    model: Pipeline Object
    model_filepath: Destination for the export
    
    OUTPUT
    The exported model as a pickle file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
