import numpy as np
import pandas as pd
import sys
import datetime
import json
import progressbar
import deepdish as dd
import random
import itertools
import utils
from base_recommender import BaseRecommender


class MPCFModel(BaseRecommender):

    def __init__(self, ratings=None, config=None):
        BaseRecommender.__init__(self, ratings, config)

        self.P = None
        self.Q = None
        self.W = None
        self.b_i = None
        self.B_u = None
        self.X = None
        self.avg_train_rating = None

        if ratings is not None and config is not None:
            nb_users = len(self.users)
            nb_movies = len(self.movies)
            nb_latent_f = config['nb_latent_f']
            nb_user_pref = config['nb_user_pref']
            nb_d2v_features = config['nb_d2v_features'] if 'nb_d2v_features' in config else None
            scale = config['init_params_scale'] if 'init_params_scale' in config else 0.001
            params = self._init_params(nb_users, nb_movies, nb_latent_f, nb_user_pref, nb_d2v_features, scale=scale)
            self._set_params(params)

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

    def _get_params(self):
        return {'P': self.P, 'Q': self.Q, 'W': self.W, 'b_i': self.b_i, 'B_u': self.B_u,
                'avg_train_rating': self.avg_train_rating, 'X': self.X}

    def _set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.W = params['W']
        self.b_i = params['b_i']
        self.B_u = params['B_u']
        self.avg_train_rating = params['avg_train_rating'] if 'avg_train_rating' in params else None

        self.X = params['X'] if 'X' in params else None

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

    def fit(self, train, val=None, test=None, d2v_model=None, si_model=None, zero_sampler=None):

        config = self.config
        lr = config['lr']
        si_lr = config['si_lr'] if 'si_lr' in config else None
        si_lr_delta_qi = config['si_lr_delta_qi'] if 'si_lr_delta_qi' in config else None
        reg_lambda = config['reg_lambda']

        if config['use_avg_rating']:
            self.avg_train_rating = train['rating'].mean()
        else:
            self.avg_train_rating = 0

        verbose = 'verbose' in config and config['verbose'] > 0

        train_rmse = []
        val_rmse = []
        feature_rmse = []

        if verbose:
            print "Start training ..."

        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}, si_lr {}, si_lr_delta_qi {}".format(epoch, lr, si_lr, si_lr_delta_qi)

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
                rating_errors.append(float(rating_error))

                # update parameters
                self.b_i[i] = b_i + lr * (rating_error - reg_lambda * b_i)
                self.B_u[u,local_pref] = B_ut + lr * (rating_error - reg_lambda * B_ut)
                self.P[u,:] = P_u + lr * (rating_error * Q_i - reg_lambda * P_u)
                self.W[u,local_pref,:] = W_ut + lr * (rating_error * Q_i - reg_lambda * W_ut)

                # side information model - predict feature vector, calculate feature vector error
                if d2v_model is not None and si_model is not None and 'si_model' in config and config['si_model']:
                    feature = d2v_model.docvecs['{}.txt'.format(imdb_id)]
                    qi_reshaped = np.reshape(Q_i, (1, -1))

                    feature_loss, delta_qi = si_model.gradient_step(qi_reshaped, feature, si_lr)

                    feature_losses.append(float(feature_loss))

                    # update parameters
                    self.Q[i,:] = Q_i + lr * (rating_error * (P_u + W_ut) - reg_lambda * Q_i) - si_lr_delta_qi * delta_qi

                else:
                    self.Q[i,:] = Q_i + lr * (rating_error * (P_u + W_ut) - reg_lambda * Q_i)

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

            if 'si_lr_decay' in config:
                si_lr *= (1.0 - config['si_lr_decay'])

            if 'si_lr_delta_qi_decay' in config:
                si_lr_delta_qi *= (1.0 - config['si_lr_delta_qi_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            if verbose:
                print "Train RMSE:", current_rmse

            if d2v_model is not None and 'si_model' in config and config['si_model']:
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
