import pandas as pd
import numpy as np
from mrec import load_fast_sparse_matrix
from mrec.item_similarity.recommender import ItemSimilarityRecommender
from mrec.item_similarity.slim import SLIM
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDRegressor
from calc_metrics import run_eval
import datetime
import json

from utils import binarize_ratings
from zero_sampler import ZeroSampler


class SLIM(ItemSimilarityRecommender):
    def __init__(self, ratings=None, config=None):
        self.ratings = ratings
        self.config = config

        self.users = None
        self.movies = None

        self.intercepts = None

        if config:
            self.ignore_negative_weights = config['ignore_negative_weights']
            self.model = SGDRegressor(penalty='elasticnet', n_iter=config['nb_epochs'],
                                      fit_intercept=config['fit_intercept'], alpha=config['alpha'],
                                      l1_ratio=config['l1_ratio'])
            self.intercepts = {}

        self.sparse_matrix = None
        if ratings is not None:
            self._create_lookup_tables(ratings)

    def _create_lookup_tables(self, ratings):
        users_unique = ratings['user_id'].unique()
        nb_users = len(users_unique)
        self.users = dict(zip(users_unique, range(nb_users)))  # lookup table for user_id to zero indexed number

        movies_unique = ratings['movie_id'].unique()
        nb_movies = len(movies_unique)
        self.movies = dict(zip(movies_unique, range(nb_movies)))  # lookup table for user_id to zero indexed number

    def _df_to_sparse_matrix(self, ratings):
        copy = ratings.copy()

        def replace_user_id(df):
            df['user_id'] = self.users[df['user_id'].unique()[0]]
            return df

        def replace_movie_id(df):
            df['movie_id'] = self.movies[df['movie_id'].unique()[0]]
            return df

        copy = copy.groupby('user_id').apply(lambda df: replace_user_id(df))
        copy = copy.groupby('movie_id').apply(lambda df: replace_movie_id(df))

        row = copy['user_id'].values
        col = copy['movie_id'].values
        data = copy['rating'].values
        shape = (len(self.users),len(self.movies))
        return csr_matrix((data,(row,col)), shape=shape)

    def compute_similarities(self, dataset, j):
        """Compute item similarity weights for item j."""
        # zero out the j-th column of the input so we get w[j] = 0
        a = dataset.fast_get_col(j)
        dataset.fast_update_col(j, np.zeros(a.nnz))
        self.model.fit(dataset.X, a.toarray().ravel())
        # reinstate the j-th column
        dataset.fast_update_col(j, a.data)
        w = self.model.coef_
        if self.ignore_negative_weights:
            w[w < 0] = 0

        if 'fit_intercept' in self.config and self.config['fit_intercept']:
            self.intercepts[j] = self.model.intercept_
        return w

    def compute_similarities_from_vec(self, dataset, a):
        """Compute item similarity weights for out-of-dataset item vector."""
        self.model.fit(dataset.X, a)
        return self.model.coef_

    def __str__(self):
        if self.ignore_negative_weights:
            return 'SLIM({0} ignoring negative weights)'.format(self.model)
        else:
            return 'SLIM({0})'.format(self.model)

    def predict(self, user_id, movie_id):
        u = self.users[user_id]
        i = self.movies[movie_id]
        r_predict = (self.similarity_matrix[i] * self.sparse_matrix[u].T).toarray().flatten()[0]
        if 'fit_intercept' in self.config and self.config['fit_intercept']:
            r_predict += self.intercepts[i]
        return r_predict

    def predict_for_user(self, user_id, movie_ids=None):
        result = []
        if movie_ids is not None:
            it = movie_ids
        else:
            it = self.movies.iterkeys()

        for movie_id in it:
            r_predict = self.predict(user_id, movie_id)
            result.append({'movie_id': movie_id, 'r_predict': r_predict})
        return pd.DataFrame(result).sort_values('r_predict', ascending=False).reset_index(drop=True)

    def fit(self, train):
        self.sparse_matrix = self._df_to_sparse_matrix(train)
        super(self.__class__, self).fit(self.sparse_matrix)

        config = self.config
        # save model, config and history
        print "Saving model ..."
        dt = datetime.datetime.now()
        #self.save('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name']))
        with open('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'
                          .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))

