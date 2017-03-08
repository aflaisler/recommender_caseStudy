import pandas as pd
import graphlab as gl

from  sklearn.model_selection import train_test_split

def load_data():
    ratings_data_fname = "data/ratings.dat"
    ratings_df = pd.read_csv(ratings_data_fname,sep='\t')
    y = ratings_df['rating']
    X = ratings_df[['user_id','joke_id']]
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    X_train,X_test,y_train,y_test = load_table()
