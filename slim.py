import pandas as pd
import numpy as np
from mrec import load_fast_sparse_matrix
from mrec.item_similarity.recommender import ItemSimilarityRecommender
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

        self.ignore_negative_weights = config['ignore_negative_weights']

        self.model = SGDRegressor(penalty='elasticnet', n_iter=config['nb_epochs'],
                                  fit_intercept=config['fit_intercept'], alpha=config['alpha'],
                                  l1_ratio=config['l1_ratio'])
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
        for user_id, u in self.users.iteritems():
            copy.loc[copy['user_id'] == user_id, 'user_id'] = u

        for movie_id, i in self.movies.iteritems():
            copy.loc[copy['movie_id'] == movie_id, 'movie_id'] = i

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


        # save model, config and history
        print "Saving model ..."
        dt = datetime.datetime.now()
        #self.save('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name']))
        with open('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'
                          .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))


if __name__ == "__main__":
    config = {}

    config['experiment_name'] = 'slim-test_e5_zero-samp-3'

    config['nb_epochs'] = 100

    config['fit_intercept'] = True
    config['ignore_negative_weights'] = False
    config['l1_reg'] = 0.001
    config['l2_reg'] = 0.0001
    config['alpha'] = config['l1_reg'] + config['l2_reg']
    config['l1_ratio'] = config['l1_reg'] / config['alpha']

    config['run_eval'] = True
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['verbose'] = 1

    ratings = pd.read_csv('data/splits/ml-100k/ratings.csv')
    train = pd.read_csv('data/splits/ml-100k/sparse-item/0.2-train.csv')
    test = pd.read_csv('data/splits/ml-100k/sparse-item/0.2-test.csv')

    config['zero_sampling'] = False
    if config['zero_sampling']:
        config['zero_sample_factor'] = 3
        config['zero_samples_total'] = len(train) * config['zero_sample_factor']

        zero_sampler = ZeroSampler(ratings)
        zero_samples = zero_sampler.sample(config['zero_samples_total'], verbose=1)
        train = train.append(zero_samples).reset_index(drop=True)

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1

    if config['binarize']:
        train = binarize_ratings(train, threshold=config['binarize_threshold'])

    model = SLIM(ratings, config)
    model.fit(train)

    config['precision_recall_at_n'] = 20
    config['verbose'] = 1
    config['hit_threshold'] = 4
    run_eval(model, train, test, ratings, config)
