import datetime
import json
import sys

import deepdish as dd
import numpy as np
import pandas as pd
import progressbar
from gensim.models import Doc2Vec

import utils
from user_pref_model import UserPrefModel


class MFNNModel(object):

    def __init__(self, ratings=None, config=None):
        self.ratings = ratings
        self.non_rated = None
        self.config = config

        self.users = None
        self.movies = None
        self.movie_to_imdb = None

        self.P = None
        self.Q = None
        self.b_i = None

        self.avg_train_rating = None

        self.user_pref_model = None
        self.d2v_model = None

        if ratings is not None and config is not None:
            self._create_lookup_tables(ratings)
            nb_users = len(self.users)
            nb_movies = len(self.movies)
            nb_latent_f = config['nb_latent_f']
            params = self._init_params(nb_users, nb_movies, nb_latent_f)
            self.set_params(params)

    def _create_lookup_tables(self, ratings):
        users_unique = ratings['user_id'].unique()
        nb_users = len(users_unique)
        self.users = dict(zip(users_unique, range(nb_users))) # lookup table for user_id to zero indexed number

        movies_unique = ratings['movie_id'].unique()
        nb_movies = len(movies_unique)
        self.movies = dict(zip(movies_unique, range(nb_movies))) # lookup table for user_id to zero indexed number

        self.movie_to_imdb = utils.movie_to_imdb(ratings)

    def _init_params(self, nb_users, nb_movies, nb_latent_f, scale=0.001):
        P = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        Q = np.random.uniform(low=-scale, high=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        b_i = np.random.uniform(low=-scale, high=scale, size=(nb_movies, 1)) # item bias vector

        params = {'P': P, 'Q': Q, 'b_i': b_i}
        return params

    def get_params(self):
        return {'P': self.P, 'Q': self.Q, 'b_i': self.b_i, 'avg_train_rating': self.avg_train_rating}

    def set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.b_i = params['b_i']
        self.avg_train_rating = params['avg_train_rating'] if 'avg_train_rating' in params else None

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

        to_save = {'config': self.config, 'params': self.get_params(), 'users': self.users, 'movies': self.movies,
                   'movie_to_imdb': self.movie_to_imdb, 'user_pref_nn_params': self.user_pref_model.param_values,
                   'd2v_model': self.d2v_model}
        dd.io.save(filepath, to_save)

    def load(self, filepath):
        loaded = dd.io.load(filepath)
        self.config = loaded['config']
        self.set_params(loaded['params'])
        self.users = loaded['users']
        self.movies = loaded['movies']
        self.movie_to_imdb = loaded['movie_to_imdb']
        self.user_pref_model = UserPrefModel(self.config)
        self.user_pref_model.set_params(loaded['user_pref_nn_params'])
        self.d2v_model = Doc2Vec.load(self.config['d2v_model'])

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
        imdb_id = self.movie_to_imdb[movie_id]

        qi = np.reshape(self.Q[i,:], (1, -1))
        pu = np.reshape(self.P[u,:], (1, -1))
        movie_d2v = np.reshape(self.d2v_model.docvecs['{}.txt'.format(imdb_id)], (1, -1))
        r_predict = self.avg_train_rating + self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T) + self.user_pref_model.predict(qi, pu, movie_d2v)
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

    def fit(self, train, val=None, test=None, zero_sampler=None, verbose=1):

        config = self.config
        lr = config['lr']
        lr_si = config['lr_si'] if 'lr_si' in config else None
        lr_delta_qi = config['lr_delta_qi'] if 'lr_delta_qi' in config else None
        reg_lambda = config['reg_lambda']

        if zero_sampler and 'zero_samples_total' in config:
            self.avg_train_rating = np.hstack((train['rating'].values, np.zeros((config['zero_samples_total'],)))).mean()
        else:
            self.avg_train_rating = train['rating'].mean()

        train_rmse = []
        val_rmse = []
        feature_rmse = []

        print "Start training ..."
        for epoch in range(config['nb_epochs']):
            print "epoch {}, lr {}, lr_si {}, lr_delta_qi {}".format(epoch, lr, lr_si, lr_delta_qi)

            if zero_sampler and 'zero_samples_total' in config:
                print "sampling zeros ..."
                zero_samples = zero_sampler.sample(config['zero_samples_total'], verbose=verbose)
                train_for_epoch = train.append(zero_samples).reset_index(drop=True)
            else:
                train_for_epoch = train

            # shuffle train
            train_for_epoch = train_for_epoch.reindex(np.random.permutation(train_for_epoch.index))

            rating_errors = []
            feature_losses = []

            if verbose > 0:
                total = len(train_for_epoch)
                bar = progressbar.ProgressBar(max_value=total)
                bar.start()
                progress = 0

            # train / update model
            for row in train_for_epoch.itertuples():
                user_id, movie_id, rating, imdb_id = row[1], row[2], row[3], row[5]

                u = self.users[user_id]
                i = self.movies[movie_id]

                # copy parameters
                b_i = self.b_i[i].copy()
                P_u = self.P[u,:].copy()
                Q_i = self.Q[i,:].copy()

                # main model - predict rating and calc rating error
                rating_mf = self.avg_train_rating + b_i + np.dot(P_u, Q_i.T)

                rating_nn_target = float(rating - rating_mf)

                if self.user_pref_model is not None:

                    qi_reshaped = np.reshape(Q_i, (1, -1))
                    pu_reshaped = np.reshape(P_u, (1, -1))
                    movie_d2v = np.reshape(self.d2v_model.docvecs['{}.txt'.format(imdb_id)], (1, -1))

                    rating_nn, loss, dQi, dPu = self.user_pref_model.gradient_step(qi_reshaped, pu_reshaped, movie_d2v, rating_nn_target, lr_si)
                    feature_losses.append(float(loss))
                rating_predict = rating_mf + float(rating_nn)

                rating_error = rating - rating_predict
                rating_errors.append(float(rating_error))

                # update parameters
                self.b_i[i] = b_i + lr * (rating_error - reg_lambda * b_i)
                self.P[u,:] = P_u + lr * (rating_error * Q_i - reg_lambda * P_u) - lr_delta_qi * dPu
                self.Q[i, :] = Q_i + lr * (rating_error * (P_u) - reg_lambda * Q_i) - lr_delta_qi * dQi

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

            if 'lr_delta_qi_decay' in config:
                lr_delta_qi *= (1.0 - config['lr_delta_qi_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            print "Train RMSE:", current_rmse

            if self.user_pref_model is not None:
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
        history = {'train_rmse': train_rmse, 'val_rmse': val_rmse, 'feature_rmse': feature_rmse, 'test_rmse': test_rmse}

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
