import itertools
import random

import numpy as np
import pandas as pd
import progressbar

import utils


class ZeroSampler(object):
    def __init__(self, ratings):
        self.ratings = ratings
        self.all_comb = None
        self.rated = None
        self.non_rated = None
        self.movie_to_imdb = utils.movie_to_imdb(ratings)
        self._init()

    def _init(self):
        movie_ids = self.ratings['movie_id'].unique()
        user_ids = self.ratings['user_id'].unique()
        self.all_comb = np.fromiter(itertools.product(user_ids, movie_ids), dtype=[('u', np.int), ('m', np.int)])
        self.rated = np.zeros((len(self.ratings),), dtype=[('u', np.int), ('m', np.int)])
        for i, row in enumerate(self.ratings.itertuples()):
            user_id, movie_id = row[1], row[2]
            self.rated[i] = (user_id, movie_id)

        self.non_rated = np.setdiff1d(self.all_comb, self.rated, assume_unique=True)

    def sample(self, nb_samples, non_rated_only=True, verbose=1):
        if non_rated_only:
            samples = random.sample(self.non_rated, nb_samples)
        else:
            samples = random.sample(self.all_comb, nb_samples)

        if verbose > 0:
            progress = 0
            bar = progressbar.ProgressBar(max_value=nb_samples)
            bar.start()

        result = []
        for i, (user_id, movie_id) in enumerate(samples):
            imdb_id = self.movie_to_imdb[movie_id]
            result.append({'user_id': user_id, 'movie_id': movie_id, 'rating': 0, 'timestamp': 0, 'imdb_id': imdb_id})
            if verbose > 0:
                progress += 1
                bar.update(progress)

        if verbose > 0:
            bar.finish()

        return pd.DataFrame(result, columns=['user_id', 'movie_id', 'rating', 'timestamp', 'imdb_id'])
