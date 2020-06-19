"""
Preprocessing data
Disaster Response Pipeline Project

Arguments:
    1) Path to a CSV file containing disaster messages (disaster_messages.csv)
    2) Path to a CSV file containing categories for the messages in point A (disaster_categories.csv)
    3) SQLite database to upload the cleaned data (DisasterResponse.db)
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths
    
    INPUT
    messages_filepath: Link to the messages file
    categories_filepath: Link to the categories file
    
    OUTPUT
    df: pandas DataFrame with the cleaned data from the inputs
    """
    
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages,categories,on='id')

    return df

def clean_data(df):
    """Clean the DataFrame
    
    INPUT
    df: Dataframe to be cleaned
    
    OUTPUT
    df: Cleaned dataframe - reformated and cleaned categories; removed duplicates; removed unusable features;
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # rename the columns of `categories`
    categories.columns = row.apply(lambda x:x[:-2])
    
    # convert the values in categories to numbers
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)    

    # drop the original categories column from `df`
    df = df.drop('categories', 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates()
    
    # remove child_alone as it has values of 0 only, hence doesn't have a meaningful contribution to the Machine Learning algorithm
    df = df.drop(['child_alone'],axis=1)
    
    # Given that only 188 rows have values of 2 in the related column, this is considered as data error and the rows will be omitted. 
    df = df[df.related != 2]
    
    return df

def save_data(df, database_filename):
    """Upload the DataFrame to a database
    
    INPUT
    df: Cleaned Dataframe
    database_filename: Path to the database to upload the data
    
    OUTPUT
    df: The function doesn't have outputs. It saves the Dataframe to the table
    """ 
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('project_disasters', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
