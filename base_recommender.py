import pandas as pd
import deepdish as dd
import sys
import utils


class BaseRecommender(object):
    def __init__(self, ratings=None, config=None):
        self.ratings = ratings
        self.non_rated = None
        self.config = config

        self.users = None
        self.movies = None
        self.movie_to_imdb = None

        if ratings is not None:
            self._create_lookup_tables(ratings)

    def _create_lookup_tables(self, ratings):
        users_unique = ratings['user_id'].unique()
        nb_users = len(users_unique)
        self.users = dict(zip(users_unique, range(nb_users)))  # lookup table for user_id to zero indexed number

        movies_unique = ratings['movie_id'].unique()
        nb_movies = len(movies_unique)
        self.movies = dict(zip(movies_unique, range(nb_movies)))  # lookup table for user_id to zero indexed number

        self.movie_to_imdb = utils.movie_to_imdb(ratings)

    def _get_params(self):
        pass

    def _set_params(self, params):
        pass

    def save(self, filepath, overwrite=False):
        """
        Save parameters; code from Keras
        """
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? '
                                  '[y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        to_save = {'config': self.config, 'params': self._get_params(), 'users': self.users, 'movies': self.movies,
                   'movie_to_imdb': self.movie_to_imdb}
        dd.io.save(filepath, to_save)

    def load(self, filepath):
        loaded = dd.io.load(filepath)
        self.config = loaded['config']
        self.users = loaded['users']
        self.movies = loaded['movies']
        self.movie_to_imdb = loaded['movie_to_imdb']
        self._set_params(loaded['params'])
        return loaded

    def predict(self, user_id, movie_id):
        pass

    def predict_for_user(self, user_id, movie_ids=None):
        result = []
        if movie_ids is not None:
            it = movie_ids
        else:
            it = self.movies.iterkeys()

        for movie_id in it:
            r_predict = self.predict(user_id, movie_id)
            result.append({'user_id': user_id, 'movie_id': movie_id, 'r_predict': r_predict})
        return pd.DataFrame(result).sort_values('r_predict', ascending=False).reset_index(drop=True)

    def predict_for_movie(self, movie_id, user_ids=None):
        result = []
        if user_ids is not None:
            it = user_ids
        else:
            it = self.users.iterkeys()

        for user_id in it:
            r_predict = self.predict(user_id, movie_id)
            result.append({'user_id': user_id, 'movie_id': movie_id, 'r_predict': r_predict})
        return pd.DataFrame(result).sort_values('r_predict', ascending=False).reset_index(drop=True)
