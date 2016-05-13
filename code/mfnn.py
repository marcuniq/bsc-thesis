import datetime
import json
import sys

import deepdish as dd
import numpy as np
import pandas as pd
import progressbar
from gensim.models import Doc2Vec

import utils
from base_recommender import BaseRecommender
from user_pref_model import UserPrefModel


class MFNNModel(BaseRecommender):

    def __init__(self, ratings=None, config=None, user_pref_model=None, d2v_model=None):
        BaseRecommender.__init__(self, ratings, config)

        self.P = None
        self.Q = None
        self.b_i = None

        self.avg_train_rating = None

        self.user_pref_model = user_pref_model
        self.d2v_model = d2v_model

        if ratings is not None and config is not None:
            nb_users = len(self.users)
            nb_movies = len(self.movies)
            nb_latent_f = config['nb_latent_f']
            params = self._init_params(nb_users, nb_movies, nb_latent_f)
            self._set_params(params)

    def _init_params(self, nb_users, nb_movies, nb_latent_f, scale=0.001):
        P = np.random.uniform(low=-scale, high=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        Q = np.random.uniform(low=-scale, high=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        b_i = np.random.uniform(low=-scale, high=scale, size=(nb_movies, 1)) # item bias vector

        params = {'P': P, 'Q': Q, 'b_i': b_i}
        return params

    def _get_params(self):
        return {'P': self.P, 'Q': self.Q, 'b_i': self.b_i, 'avg_train_rating': self.avg_train_rating,
                'user_pref_nn_params': self.user_pref_model.param_values}

    def _set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.b_i = params['b_i']
        self.avg_train_rating = params['avg_train_rating'] if 'avg_train_rating' in params else None

        if self.user_pref_model is None and 'user_pref_nn_params' in params:
            self.user_pref_model = UserPrefModel(self.config)
            self.user_pref_model.set_params(params['user_pref_nn_params'])
        if self.d2v_model is None and 'd2v_model' in self.config:
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

    def fit(self, train, val=None, test=None, zero_sampler=None, verbose=1):

        config = self.config
        lr = config['lr']
        user_pref_lr = config['user_pref_lr']
        user_pref_lambda_grad = config['user_pref_lambda_grad']
        reg_lambda = config['reg_lambda']

        if config['use_avg_rating']:
            self.avg_train_rating = train['rating'].mean()
        else:
            self.avg_train_rating = 0

        verbose = 'verbose' in config and config['verbose'] > 0

        # AdaGrad
        if 'adagrad' in config and config['adagrad']:
            ada_cache_b_i = np.zeros_like(self.b_i)
            ada_cache_P = np.zeros_like(self.P)
            ada_cache_Q = np.zeros_like(self.Q)
            ada_eps = config['ada_eps']

        train_rmse = []
        val_rmse = []
        feature_rmse = []

        if verbose:
            print "Start training ..."
        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}, user_pref_lr {}".format(epoch, lr, user_pref_lr)

            if zero_sampler and 'zero_samples_total' in config:
                if verbose:
                    print "sampling zeros ..."
                zero_samples = zero_sampler.sample(config['zero_samples_total'], verbose=verbose)
                train_for_epoch = train.append(zero_samples).reset_index(drop=True)
            else:
                train_for_epoch = train

            # shuffle train
            train_for_epoch = train_for_epoch.reindex(np.random.permutation(train_for_epoch.index))

            rating_errors = []
            feature_losses = []

            if verbose:
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

                # neural network model
                qi_reshaped = np.reshape(Q_i, (1, -1))
                pu_reshaped = np.reshape(P_u, (1, -1))
                movie_d2v = np.reshape(self.d2v_model.docvecs['{}.txt'.format(imdb_id)], (1, -1))

                rating_nn, loss, nn_dQ_i, nn_dP_u = self.user_pref_model.gradient_step(qi_reshaped, pu_reshaped, movie_d2v, rating_nn_target, user_pref_lr)
                feature_losses.append(float(loss))

                rating_predict = rating_mf + float(rating_nn)

                rating_error = rating - rating_predict
                rating_errors.append(float(rating_error))

                # calc gradients
                db_i = rating_error - reg_lambda * b_i
                dP_u = rating_error * Q_i - reg_lambda * P_u - user_pref_lambda_grad * nn_dP_u.flatten()
                dQ_i = rating_error * P_u - reg_lambda * Q_i - user_pref_lambda_grad * nn_dQ_i.flatten()

                # update AdaGrad caches and parameters
                if 'adagrad' in config and config['adagrad']:
                    ada_cache_b_i[i] += db_i ** 2
                    ada_cache_P[u, :] += dP_u ** 2
                    ada_cache_Q[i, :] += dQ_i ** 2

                    # update parameters
                    self.b_i[i] = b_i + lr * db_i / (np.sqrt(ada_cache_b_i[i]) + ada_eps)
                    self.P[u, :] = P_u + lr * dP_u / (np.sqrt(ada_cache_P[u, :]) + ada_eps)
                    self.Q[i, :] = Q_i + lr * dQ_i / (np.sqrt(ada_cache_Q[i, :]) + ada_eps)
                else:
                    self.b_i[i] = b_i + lr * db_i
                    self.P[u, :] = P_u + lr * dP_u
                    self.Q[i, :] = Q_i + lr * dQ_i

                # update progess bar
                if verbose:
                    progress += 1
                    bar.update(progress)

            if verbose:
                bar.finish()

            # lr decay
            if 'lr_decay' in config:
                lr *= (1.0 - config['lr_decay'])
            elif 'lr_power_t' in config:
                lr = config['lr'] / pow(epoch+1, config['lr_power_t'])

            if 'user_pref_lr_decay' in config:
                user_pref_lr *= (1.0 - config['user_pref_lr_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            if verbose:
                print "Train RMSE:", current_rmse

            if self.user_pref_model is not None:
                current_feature_rmse = np.sqrt(np.mean(feature_losses))
                feature_rmse.append(current_feature_rmse)
                if verbose:
                    print "Feature RMSE:", current_feature_rmse

            # validation
            if val is not None and 'val' in config and config['val']:
                val_errors = self.test(val)

                # report error
                current_val_rmse = np.sqrt(np.mean(np.square(val_errors)))
                val_rmse.append(current_val_rmse)
                if verbose:
                    print "Validation RMSE:", current_val_rmse

        # report error on test set
        test_rmse = []
        if test is not None and 'test' in config and config['test']:
            test_errors = self.test(test)

            # report error
            test_rmse = np.sqrt(np.mean(np.square(test_errors)))
            if verbose:
                print "Test RMSE:", test_rmse

        # history
        history = {'train_rmse': train_rmse, 'val_rmse': val_rmse, 'feature_rmse': feature_rmse, 'test_rmse': test_rmse}
        return history
