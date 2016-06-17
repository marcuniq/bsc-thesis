import itertools
import random

import numpy as np
import pandas as pd
import progressbar


class ZeroSampler(object):
    def __init__(self, ratings, user_key='user_id', item_key='movie_id'):
        self.ratings = ratings
        self.user_key = user_key
        self.item_key = item_key
        self.all_comb = None
        self.rated = None
        self.non_rated = None
        self._init()

    def _init(self):
        item_ids = self.ratings[self.item_key].unique()
        user_ids = self.ratings[self.user_key].unique()
        self.all_comb = np.fromiter(itertools.product(user_ids, item_ids), dtype=[('u', np.int), ('i', np.int)])
        self.rated = np.zeros((len(self.ratings),), dtype=[('u', np.int), ('i', np.int)])
        for i, row in enumerate(self.ratings.itertuples()):
            user_id, item_id = row[1], row[2]
            self.rated[i] = (user_id, item_id)

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
        for i, (user_id, item_id) in enumerate(samples):
            result.append({self.user_key: user_id, self.item_key: item_id, 'rating': 0, 'timestamp':0, 'imdb_id':0})
            if verbose > 0:
                progress += 1
                bar.update(progress)

        if verbose > 0:
            bar.finish()

        return pd.DataFrame(result, columns=[self.user_key, self.item_key, 'rating', 'timestamp', 'imdb_id'])
