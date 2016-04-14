import numpy as np
import pandas as pd
import sys
import datetime
import json
from gensim.models.doc2vec import Doc2Vec
import progressbar
import deepdish as dd

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
        self.X = None
        self.g = None
        self.avg_train_rating = None

        if ratings is not None and config is not None:
            self._create_lookup_tables(ratings)
            nb_users = len(self.users)
            nb_movies = len(self.movies)
            nb_latent_f = config['nb_latent_f']
            nb_user_pref = config['nb_user_pref']
            nb_d2v_features = config['nb_d2v_features'] if 'nb_d2v_features' in config else None
            params = self._init_params(nb_users, nb_movies, nb_latent_f, nb_user_pref, nb_d2v_features)
            self.set_params(params)

    def _create_lookup_tables(self, ratings):
        users_unique = ratings['user_id'].unique()
        nb_users = len(users_unique)
        self.users = dict(zip(users_unique, range(nb_users))) # lookup table for user_id to zero indexed number

        movies_unique = ratings['movie_id'].unique()
        nb_movies = len(movies_unique)
        self.movies = dict(zip(movies_unique, range(nb_movies))) # lookup table for user_id to zero indexed number

    def _init_params(self, nb_users, nb_movies, nb_latent_f, nb_user_pref, nb_d2v_features=None, scale=0.001):
        P = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        Q = np.random.uniform(low=-scale, high=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        W = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_user_pref, nb_latent_f)) # user latent factor tensor
        b_i = np.random.uniform(low=-scale, high=scale, size=(nb_movies, 1)) # item bias vector
        B_u = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_user_pref)) # user-interest bias matrix

        params = {'P': P, 'Q': Q, 'W': W, 'b_i': b_i, 'B_u': B_u}
        if nb_d2v_features is not None:
            X = np.random.uniform(low=-scale, high=scale, size=(nb_d2v_features, nb_latent_f+1)) # side info weight matrix
            params['X'] = X
        return params

    def get_params(self):
        return {'P': self.P, 'Q': self.Q, 'W': self.W, 'b_i': self.b_i, 'B_u': self.B_u,
                'avg_train_rating': self.avg_train_rating, 'X': self.X}

    def set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.W = params['W']
        self.b_i = params['b_i']
        self.B_u = params['B_u']
        self.avg_train_rating = params['avg_train_rating'] if 'avg_train_rating' in params else None

        self.X = params['X'] if 'X' in params else None

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

        to_save = {'config': self.config, 'params': self.get_params(), 'users': self.users, 'movies': self.movies}
        dd.io.save(filepath, to_save)

    def load(self, filepath):
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

            r_predict = self.predict(user_id, movie_id)

            test_error = rating - r_predict
            test_errors.append(test_error)
        return test_errors

    def predict(self, user_id, movie_id):
        u = self.users[user_id]
        i = self.movies[movie_id]
        local_pref, local_pref_score = self._get_local_pref(u, i)
        r_predict = self.avg_train_rating + self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T) + local_pref_score
        return r_predict

    def predict_for_user(self, user_id, movie_ids=None):
        u = self.users[user_id]
        result = pd.DataFrame()

        if movie_ids is not None:
            it = movie_ids
        else:
            it = self.movies.iterkeys()

        for movie_id in it:
            i = self.movies[movie_id]
            local_pref, local_pref_score = self._get_local_pref(u, i)
            r_predict = self.avg_train_rating + self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T) + local_pref_score
            result = result.append(pd.DataFrame({'movie_id': movie_id, 'r_predict': r_predict}))
        return result.sort_values('r_predict', ascending=False).reset_index(drop=True)

    def fit(self, train, val=None, test=None, d2v_model=None, verbose=1):

        config = self.config
        lr = config['lr']
        lr_si = config['lr_si'] if 'lr_si' in config else None
        lambda_bi = config['lambda_bi']
        lambda_p = config['lambda_p']

        self.avg_train_rating = train['rating'].mean()

        train_rmse = []
        val_rmse = []
        feature_rmse = []

        print "Start training ..."
        for epoch in range(config['nb_epochs']):
            print "epoch {}, lr {}".format(epoch, lr)

            # shuffle train
            train = train.reindex(np.random.permutation(train.index))

            rating_errors = []
            feature_losses = []

            if verbose > 0:
                total = len(train)
                bar = progressbar.ProgressBar(max_value=total)
                bar.start()
                progress = 0

            # train / update model
            for row in train.itertuples():
                user_id, movie_id, rating, imdb_id = row[1], row[2], row[3], row[5]

                u = self.users[user_id]
                i = self.movies[movie_id]

                local_pref, local_pref_score = self._get_local_pref(u, i)

                # copy parameters
                b_i = self.b_i[i].copy()
                B_ut = self.B_u[u,local_pref].copy()
                P_u = self.P[u,:].copy()
                Q_i = self.Q[i,:].copy()
                W_ut = self.W[u,local_pref,:].copy()

                # main model - predict rating and calc rating error
                rating_predict = self.avg_train_rating + b_i + np.dot(P_u, Q_i.T) + local_pref_score

                rating_error = rating - rating_predict
                rating_errors.append(rating_error)

                # update parameters
                self.b_i[i] = b_i + lr * (rating_error - lambda_bi * b_i)
                self.B_u[u,local_pref] = B_ut + lr * (rating_error - lambda_p * B_ut)
                self.P[u,:] = P_u + lr * (rating_error * Q_i - lambda_p * P_u)
                self.W[u,local_pref,:] = W_ut + lr * (rating_error * Q_i - lambda_p * W_ut)

                # side information model - predict feature vector, calculate feature vector error
                if d2v_model is not None and 'si_model' in config and config['si_model']:
                    feature_predict = np.dot(self.X, np.hstack([1, Q_i]).T) # vector

                    feature = d2v_model.docvecs['{}.txt'.format(imdb_id)]
                    feature_error = feature - feature_predict
                    deltaX = np.dot(feature_error.reshape((-1,1)), np.hstack([1, Q_i]).reshape((1, -1)))
                    deltaQ_i = np.dot(self.X[:,1:].T, feature_error) # without bias
                    feature_loss = np.sum(np.square(feature_error))
                    feature_losses.append(feature_loss)

                    # update parameters
                    self.Q[i,:] = Q_i + lr * (rating_error * (P_u + W_ut) - lambda_p * Q_i + deltaQ_i)
                    self.X = self.X + lr_si * (deltaX - lambda_p * self.X)
                else:
                    self.Q[i,:] = Q_i + lr * (rating_error * (P_u + W_ut) - lambda_p * Q_i)

                # update progess bar
                if verbose > 0:
                    progress += 1
                    bar.update(progress)

            if bar:
                bar.finish()

            # lr decay
            if 'lr_decay' in config:
                lr *= (1.0 - config['lr_decay'])

            if 'lr_si_decay' in config:
                lr_si *= (1.0 - config['lr_si_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            print "Train RMSE:", current_rmse

            if d2v_model is not None and 'si_model' in config and config['si_model']:
                current_feature_rmse = np.sqrt(np.mean(feature_losses))
                feature_rmse.append(current_feature_rmse)
                print "Feature RMSE:", current_feature_rmse

            # validation
            if val is not None and 'val' in config and config['val']:
                val_errors = self.test(val)

                # report error
                current_val_rmse = np.sqrt(np.mean(np.square(val_errors)))
                val_rmse.append(current_val_rmse)
                print "Validation RMSE:", current_val_rmse

            # save
            if 'save_on_epoch_end' in config and config['save_on_epoch_end']:
                print "Saving model ..."
                dt = datetime.datetime.now()
                self.save('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}.h5'.format(dt, config['experiment_name'], epoch))

        # report error on test set
        test_rmse = []
        if test is not None and 'test' in config and config['test']:
            test_errors = self.test(test)

            # report error
            test_rmse = np.sqrt(np.mean(np.square(test_errors)))
            print "Test RMSE:", test_rmse

        # history
        history = {'train_rmse': train_rmse, 'val_rmse': val_rmse, 'featur_rmse': feature_rmse, 'test_rmse': test_rmse}

        # save model, config and history
        print "Saving model ..."
        dt = datetime.datetime.now()
        self.save('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name']))
        with open('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'
                      .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))
        with open('mpcf-models/{:%Y-%m-%d_%H.%M.%S}_{}_history.json'
                              .format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(history))

        return history


if __name__ == "__main__":
    config = {'lr': 0.001, 'lr_decay': 5e-4, 'lambda_bi': 0.06, 'lambda_p': 0.06, 'nb_latent_f': 128, 'nb_user_pref': 2,
              'nb_epochs': 50, 'val': True, 'test': True,
              'save_on_epoch_end': False, 'train_test_split': 0.8, 'train_val_split': 0.9}

    # ratings_path = 'data/ml-1m/processed/ratings.csv'
    # movies_path = 'data/ml-1m/processed/movies-enhanced.csv'
    # all_subs_path = 'data/subs/all.txt'
    # ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    # train, test = get_train_test_split(ratings, train_size=config['train_test_split'], sparse_item=False)

    ratings = pd.read_csv('data/splits/ratings.csv')
    train = pd.read_csv('data/splits/0.8-train.csv')
    train, val = get_train_test_split(train, train_size=config['train_val_split'], sparse_item=False)
    test = pd.read_csv('data/splits/0.8-test.csv')

    config['experiment_name'] = 'si_kabbur-best_e50'
    side_info_model = True

    d2v_model = None

    if side_info_model:
        config['d2v_model'] = 'doc2vec-models/2016-04-11_17.48.40_100e_lr0.01_window8_neg4'
        #config['d2v_model'] = 'doc2vec-models/doc2vec-model_stopwords-removed'
        d2v_model = Doc2Vec.load(config['d2v_model'])
        config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
        config['si_model'] = True
        config['lr_si'] = 0.005
        config['lr_si_decay'] = 2e-2

    model = MPCFModel(ratings, config)
    model.fit(train, val=val, test=test, d2v_model=d2v_model)

