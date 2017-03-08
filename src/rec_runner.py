import pandas as pd
import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def score(df_actual, df_prediction):
    """Look at 5% of most highly predicted jokes for each user.
    Return the average actual rating of those jokes.
    """
    #sample = pd.read_csv('data/sample_submission.csv')

    df = pd.concat([df_prediction, df_actual], axis=1)
    g = df.groupby('user_id')

    top_5 = g.rating.apply(
        lambda x: x >= x.quantile(.95)
    )

    return df_actual[top_5==1].mean()['rating']

def load_data(filename, sample):
    sf = gl.SFrame(filename, format = 'tsv')
    df_sample = pd.read_csv(sample)
    sf_sample = gl.SFrame(df_sample)
    return sf, df_sample, sf_sample

def rec_factorization(sf):
    mf = gl.recommender.factorization_recommender.create(observation_data=sf,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='als',
                                                     regularization = 0,
                                                     verbose = False)

    return mf


if __name__ == "__main__":
    valid_fname = "../data/validation_data.csv"
    ratings_data_fname = "../data/ratings.dat"
    output_fname = "../data/test_ratings.csv"
    sf, df_valid, sf_valid = load_data(ratings_data_fname, valid_fname)

    train, test = gl.recommender.util.random_split_by_user(sf, 'user_id', 'joke_id')

    df_actual = pd.DataFrame()
    df_pred = pd.DataFrame()

    df_actual['user_id'] = sf_valid['user_id']
    df_actual['joke_id'] = sf_valid['joke_id']

    df_actual['rating'] = sf_valid['rating']

    # i = range(1, 100, 1)
    # for a in i:
    #     m = rec_factorization(sf, a)
    #     df_pred['pred_rating'] = m.predict(sf_valid)
    #     score = score(df_actual, df_pred)
    #     #rmse = np.sqrt(mean_squared_error(sf_valid['rating'], df_pred['pred_rating']))
    #     print 'Score at %s : %s' %(a, score)
    # m = rec_factorization(sf)
    # df_pred['pred_rating'] = m.predict(sf_valid)
    # my_score = score(df_actual, df_pred)
    #max_score = score(df_actual, df_actual)
    #rmse = np.sqrt(mean_squared_error(sf_valid['rating'], df_pred['pred_rating']))
    # print 'Score for factorization_recommender: %s' %my_score

    # sample_sub.rating = rec_engine.predict(for_prediction)
    # sample_sub.to_csv(output_fname, index=False)

    # m_r = gl.recommender.create(sf, user_id='user_id', item_id='joke_id',target="rating")
    # df_pred['pred_rating'] = m_r.predict(sf_valid)
    # my_score = score(df_actual, df_pred)
    # print 'Score for recommender: %s' %my_score
    #
    # m_fr_reg0 = gl.recommender.factorization_recommender.create(observation_data=sf,
    #                                               user_id="user_id",
    #                                               item_id="joke_id",
    #                                               target='rating',
    #                                               solver='als',
    #                                               side_data_factorization=False,
    #                                               regularization=0,random_seed=0)
    # df_pred['pred_rating'] = m_r.predict(sf_valid)
    # my_score = score(df_actual, df_pred)
    # print 'Score for factorization_recommender: %s' %my_score
    #
    # m_rfr_reg0 = gl.recommender.ranking_factorization_recommender.create(observation_data=sf,
    #                                               user_id="user_id",
    #                                               item_id="joke_id",
    #                                               target='rating',
    #                                               solver='auto',
    #                                               side_data_factorization=False,
    #                                               regularization=0,random_seed=0)
    # df_pred['pred_rating'] = m_rfr_reg0.predict(sf_valid)
    # my_score = score(df_actual, df_pred)
    # print 'Score for ranking_factorization_recommender: %s' %my_score

    m_itemsim = gl.item_similarity_recommender.create(sf,user_id="user_id",
                                                  item_id="joke_id",only_top_k=5,target = "rating")

    df_pred['pred_rating'] = m_itemsim.predict(sf_valid)
    my_score = score(df_actual, df_pred)
    print 'Score for m_itemsim: %s' %my_score
