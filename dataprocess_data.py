import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load data
    Input
    messages_filepath: path of the messages file
    categories_filepath: path of the categories file
    """
    # read files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the two files
    df = pd.merge(messages,categories,on='id')

    return df
    
def clean_data(df):
    """Split categories into separate category columns
    Input: df (dataframe)
    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # get names of each column
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        # categories[column] = categories[column].str[-1]
        categories[column] = categories[column].apply(lambda x:str(x)[-1]).values    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Replace categories column in df with new category columns
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)

    # remove duplicates
    df.drop_duplicates(subset='message',inplace=True)
    
    return df
       
def save_data(df, database_filename):
    """Save data to database
    Input
    df: dataframe to be saved
    database_filename: name of the database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_messages', engine, index=False)
    return


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
