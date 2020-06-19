import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('project_disasters', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_percentage = round(100*genre_counts/genre_counts.sum(), 1)
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    
    category_news_counts = df[df['genre']=='news'].iloc[:,4:].sum().sort_values(ascending=False)
    category_news_names = list(category_news_counts.index)
    
    category_direct_counts = df[df['genre']=='direct'].iloc[:,4:].sum().sort_values(ascending=False)
    category_direct_names = list(category_direct_counts.index)
    
    category_social_counts = df[df['genre']=='social'].iloc[:,4:].sum().sort_values(ascending=False)
    category_social_names = list(category_social_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Graph 1
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.5,
                "name": "Genre",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "Count and Percent of Messages by Genre"
            }
        },
        
        #Graph 2
        {
            'data': [
                {
                "type": "bar",
                "x": category_names ,
                "y": category_counts
                }
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                    
                }
            }
        },
        
        #Graph 3
        {
            'data': [
                {
                "type": "bar",
                "x": category_news_names,
                "y": category_news_counts
                }
            ],

            'layout': {
                'title': 'Distribution of Categories for news genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                    
                }
            }
        },
        #Graph 4
        {
            'data': [
                {
                "type": "bar",
                "x": category_direct_names ,
                "y": category_direct_counts
                }
            ],

            'layout': {
                'title': 'Distribution of Categories for direct genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                    
                }
            }
        },
        #Graph 5
        {
            'data': [
                {
                "type": "bar",
                "x": category_social_names ,
                "y": category_social_counts,
                }
            ],

            'layout': {
                'title': 'Distribution of Categories for social genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                    
                }
            }
        }
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()