import sys

import deepdish as dd
import pandas as pd


class BaseRecommender(object):
    def __init__(self, users, items, config=None):
        self.users = users
        self.items = items
        self.non_rated = None
        self.config = config

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

        to_save = {'config': self.config, 'params': self._get_params(), 'users': self.users, 'items': self.items}
        dd.io.save(filepath, to_save)

    def load(self, filepath):
        loaded = dd.io.load(filepath)
        self.config = loaded['config']
        self.users = loaded['users']
        self.items = loaded['movies'] if 'items' not in loaded else loaded['items']
        self._set_params(loaded['params'])
        return loaded

    def predict(self, user_id, item_id):
        pass

    def predict_for_user(self, user_id, item_ids=None):
        result = []
        if item_ids is not None:
            it = item_ids
        else:
            it = self.items.iterkeys()

        for item_id in it:
            r_predict = self.predict(user_id, item_id)
            result.append({'user_id': user_id, 'movie_id': item_id, 'r_predict': r_predict})
        return pd.DataFrame(result).sort_values('r_predict', ascending=False).reset_index(drop=True)

    def predict_for_movie(self, item_id, user_ids=None):
        result = []
        if user_ids is not None:
            it = user_ids
        else:
            it = self.users.iterkeys()

        for user_id in it:
            r_predict = self.predict(user_id, item_id)
            result.append({'user_id': user_id, 'movie_id': item_id, 'r_predict': r_predict})
        return pd.DataFrame(result).sort_values('r_predict', ascending=False).reset_index(drop=True)
