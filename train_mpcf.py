import numpy as np
import sys
import datetime
import json
from ratings import get_ratings, get_train_test_split


class MPCFModel(object):

    def __init__(self, ratings=None, config=None):
        self.config = config

        self.users = None
        self.movies = None

        self.P = None
        self.Q = None
        self.W = None
        self.b_i = None
        self.B_u = None
        self.avg_train_rating = None

        if ratings is not None and config is not None:
            self._create_lookup_tables(ratings)
            nb_users = len(self.users)
            nb_movies = len(self.movies)
            params = self._init_params(nb_users, nb_movies, config['nb_latent_f'], config['nb_user_pref'])
            self.set_params(params)

    def _create_lookup_tables(self, ratings):
        users_unique = ratings['user_id'].unique()
        nb_users = len(users_unique)
        self.users = dict(zip(users_unique, range(nb_users))) # lookup table for user_id to zero indexed number

        movies_unique = ratings['movie_id'].unique()
        nb_movies = len(movies_unique)
        self.movies = dict(zip(movies_unique, range(nb_movies))) # lookup table for user_id to zero indexed number

    def _init_params(self, nb_users, nb_movies, nb_latent_f, nb_user_pref, scale=0.001):
        P = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        Q = np.random.uniform(low=-scale, high=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        W = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_user_pref, nb_latent_f)) # user latent factor tensor
        b_i = np.random.uniform(low=-scale, high=scale, size=(nb_movies, 1)) # item bias vector
        B_u = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_user_pref)) # user-interest bias matrix

        return {'P': P, 'Q': Q, 'W': W, 'b_i': b_i, 'B_u': B_u}

    def get_params(self):
        return {'P': self.P, 'Q': self.Q, 'W': self.W, 'b_i': self.b_i, 'B_u': self.B_u,
                'avg_train_rating': self.avg_train_rating}

    def set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.W = params['W']
        self.b_i = params['b_i']
        self.B_u = params['B_u']

        self.avg_train_rating = params['avg_train_rating'] if 'avg_train_rating' in params else None

    def save(self, filepath, overwrite=False):
        """
        Save parameters; code from Keras
        """
        import deepdish as dd
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
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

        to_save = {'config': self.config, 'params': self.get_params(), 'users': self.users, 'movies': self.movies}
        dd.io.save(filepath, to_save)

    def load(self, filepath):
        import deepdish as dd

        loaded = dd.io.load(filepath)
        self.config = loaded['config']
        self.set_params(loaded['params'])
        self.users = loaded['users']
        self.movies = loaded['movies']

    def _get_local_pref(self, u, i):
        max_score = False
        local_pref = 0
        for t in range(self.config['nb_user_pref']):
            score = self.B_u[u,t] + np.dot(self.W[u,t,:], self.Q[i,:].T)
            if not max_score or score > max_score:
                max_score = score
                local_pref = t
        return local_pref, max_score

    def test(self, data):
        test_errors = []
        for row in data.itertuples():
            user_id, movie_id, rating = row[1], row[2], row[3]

            u = self.users[user_id]
            i = self.movies[movie_id]
            local_pref, local_pref_score = self._get_local_pref(u, i)
            r_predict = self.avg_train_rating + self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T) + local_pref_score

            test_error = rating - r_predict
            test_errors.append(test_error**2)
        return test_errors

    def fit(self, train, val=None, test=None, verbose=1):

        config = self.config
        lr = config['lr']
        lambda_bi = config['lambda_bi']
        lambda_p = config['lambda_p']

        self.avg_train_rating = train['rating'].mean()

        train_rmse = []
        val_rmse = []

        print "Start training ..."
        for epoch in range(config['nb_epochs']):
            print "epoch {}".format(epoch)

            # shuffle train
            train = train.reindex(np.random.permutation(train.index))

            errors = []
            if verbose > 0:
                total = len(train)
                point = total / 100
                increment = total / 20
                progress = 0

            # train / update model
            for row in train.itertuples():
                user_id, movie_id, rating = row[1], row[2], row[3]

                u = self.users[user_id]
                i = self.movies[movie_id]

                local_pref, local_pref_score = self._get_local_pref(u, i)
                r_predict = self.avg_train_rating + self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T) + local_pref_score

                error = rating - r_predict
                errors.append(error**2)
                self.b_i[i] = self.b_i[i] + lr * (error - lambda_bi * self.b_i[i])
                self.B_u[u,local_pref] = self.B_u[u,local_pref] + lr * (error - lambda_p * self.B_u[u,local_pref])
                self.P[u,:] = self.P[u,:] + lr * (error * self.Q[i,:] - lambda_p * self.P[u,:])
                self.Q[i,:] = self.Q[i,:] + lr * (error * self.P[u,:] - lambda_p * self.Q[i,:])
                self.W[u,local_pref,:] = self.W[u,local_pref,:] + lr * (error * self.Q[i,:] - lambda_p * self.W[u,local_pref,:])

                # update progess bar
                if verbose > 0:
                    if(progress % (5 * point) == 0):
                        sys.stdout.write("\r[" + "=" * (progress / increment) +  " " * ((total - progress)/ increment) + "]"
                                         + str(progress / point) + "%")
                        sys.stdout.flush()

                    progress = progress + 1

            print ""
            # report error
            current_rmse = np.sqrt(np.mean(errors))
            train_rmse.append(current_rmse)
            print "Train RMSE:", current_rmse

            # validation
            if val is not None and 'val' in config and config['val']:
                val_errors = self.test(val)

                # report error
                current_val_rmse = np.sqrt(np.mean(val_errors))
                val_rmse.append(current_val_rmse)
                print "Validation RMSE:", current_val_rmse

            # save
            if 'save_on_epoch_end' in config and config['save_on_epoch_end']:
                print "Saving model ..."
                dt = datetime.datetime.now()
                self.save('mpcf-models\\{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}.h5'.format(dt, config['experiment_name'], epoch))

        # report error on test set
        test_rmse = []
        if test is not None and 'test' in config and config['test']:
            test_errors = self.test(test)

            # report error
            test_rmse = np.sqrt(np.mean(test_errors))
            print "Test RMSE:", test_rmse

        # history
        history = {'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse}

        # save model, config and history
        print "Saving model ..."
        dt = datetime.datetime.now()
        self.save('mpcf-models\\{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name']))
        with open('mpcf-models\\{:%Y-%m-%d_%H.%M.%S}_{}_config.json'
                      .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))
        with open('mpcf-models\\{:%Y-%m-%d_%H.%M.%S}_{}_history.json'
                              .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(history))

        return history


if __name__ == "__main__":
    config = {'lr': 0.001, 'lambda_bi': 0.06, 'lambda_p': 0.06, 'nb_latent_f': 128, 'nb_user_pref': 2,
              'nb_epochs': 20, 'val': True, 'experiment_name': 'kabbur_best', 'save_on_epoch_end': False}
    ratings_path = 'data\\ml-1m\\processed\\ratings.csv'
    movies_path = 'data\\ml-1m\\processed\\movies-enhanced.csv'
    all_subs_path = 'data\\subs\\all.txt'
    ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    train, test = get_train_test_split(ratings, train_size=0.8, sparse_item=True)
    train, val = get_train_test_split(train, train_size=0.8, sparse_item=True)

    #model = MPCFModel()
    #model.load('mpcf-models/2016-04-01_15.43.01_test.h5')
    #error = model.test(test)

    model = MPCFModel(ratings, config)
    hist = model.fit(train, val, test)

    print hist
