import pandas as pd
import numpy as np
from mrec.sparse import fast_sparse_matrix
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDRegressor
import datetime
import json

from utils import binarize_ratings
from zero_sampler import ZeroSampler
from base_recommender import BaseRecommender


class SLIMModel(BaseRecommender):
    def __init__(self, ratings=None, config=None):
        BaseRecommender.__init__(self, ratings, config)

        self.sparse_matrix = None
        self.similarity_matrix = None
        self.intercepts = None

        if config:
            self.ignore_negative_weights = config['ignore_negative_weights']
            self.model = SGDRegressor(penalty='elasticnet', n_iter=config['nb_epochs'],
                                      fit_intercept=config['fit_intercept'], alpha=config['alpha'],
                                      l1_ratio=config['l1_ratio'])
            self.intercepts = {}

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

    def _get_params(self):
        params = {'sparse_matrix': self.sparse_matrix, 'similarity_matrix': self.similarity_matrix,
                  'intercepts': self.intercepts}
        return params

    def _set_params(self, params):
        self.sparse_matrix = params['sparse_matrix']
        self.similarity_matrix = params['similarity_matrix']
        self.intercepts = params['intercepts']

    def predict(self, user_id, movie_id):
        u = self.users[user_id]
        i = self.movies[movie_id]
        r_predict = (self.similarity_matrix[i] * self.sparse_matrix[u].T).toarray().flatten()[0]
        if 'fit_intercept' in self.config and self.config['fit_intercept']:
            r_predict += self.intercepts[i]
        return r_predict

    def fit(self, train):
        """ train is a pandas DataFrame, which has columns:
                'movie_id'
                'user_id'
                'rating'
        """
        self.sparse_matrix = self._df_to_sparse_matrix(train)

        # copied from mrec ItemSimilarityRecommender
        if not isinstance(self.sparse_matrix, fast_sparse_matrix):
            dataset = fast_sparse_matrix(self.sparse_matrix)
        num_users, num_items = dataset.shape
        # build up a sparse similarity matrix
        data = []
        row = []
        col = []
        for j in xrange(num_items):
            w = self.compute_similarities(dataset, j)
            for k, v in enumerate(w):
                if v != 0:
                    data.append(v)
                    row.append(j)
                    col.append(k)
        idx = np.array([row, col], dtype='int32')
        self.similarity_matrix = csr_matrix((data, idx), (num_items, num_items))

        config = self.config
        # save model, config and history
        if config['verbose']:
            print "Saving model ..."
        dt = datetime.datetime.now()
        self.save('slim-models/{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name']))
        with open('slim-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'
                          .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))

