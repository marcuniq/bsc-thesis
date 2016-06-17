import deepdish as dd
import lasagne
import numpy as np
import theano
import theano.tensor as T
from gensim.models import Doc2Vec

from ..utils import MovieIdToDocVec, UserIdToDocVec


class BaseSideInfoModel(object):
    def step(self, input, label, lr):
        raise NotImplementedError


class NNSideInfoModel(BaseSideInfoModel):
    """Neural Network"""
    def __init__(self, dim, reg_lambda=None, cosine_lambda=None, feature_vec_dict=None):
        self.vector_dict = feature_vec_dict

        input_vec = T.matrix(dtype=theano.config.floatX)
        label_vec = T.vector(dtype=theano.config.floatX)

        network = lasagne.layers.InputLayer(shape=(None, dim[0]), input_var=input_vec)
        for i in range(1, len(dim)-1):
            n_out = dim[i]
            network = lasagne.layers.DenseLayer(network, num_units=n_out)
        network = lasagne.layers.DenseLayer(network, num_units=dim[-1], nonlinearity=lasagne.nonlinearities.identity)
        label_predict = lasagne.layers.get_output(network)

        self.predict = theano.function([input_vec], label_predict, allow_input_downcast=True)
        self.param_values = lasagne.layers.get_all_param_values(network)
        self.network = network

        if reg_lambda is not None and feature_vec_dict is not None:
            loss = lasagne.objectives.squared_error(label_vec, label_predict).mean()
            # cosine similarity
            loss += cosine_lambda * (1.0 - (T.sum(label_vec * label_predict)) / (T.sqrt(T.sum(T.square(label_vec))) * T.sqrt(T.sum(T.square(label_predict)))))
            loss += reg_lambda * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

            d_input_vec = theano.grad(loss, input_vec)
            lr = T.scalar('lr', dtype=theano.config.floatX)

            # create parameter update expressions
            params = lasagne.layers.get_all_params(network, trainable=True)
            updates = lasagne.updates.adagrad(loss, params, lr)

            self._gradient_step = theano.function([input_vec, label_vec, lr], outputs=[loss, d_input_vec], updates=updates, allow_input_downcast=True)

    def step(self, input_vec, user_or_item_id, lr):
        label_vec = self.vector_dict[user_or_item_id]
        return self._gradient_step(input_vec, label_vec, lr)

    def set_params(self, param_values):
        params_float32 = map(lambda arr: arr.astype(np.float32), param_values)
        lasagne.layers.set_all_param_values(self.network, params_float32)


class ATSideInfoModel(BaseSideInfoModel):
    """Affine Transformation"""
    def __init__(self, dim, reg_lambda=None, cosine_lambda=None, feature_vec_dict=None):
        self.vector_dict = feature_vec_dict

        nb_latent_f = dim[0]
        nb_d2v_features = dim[-1]

        self.weights = np.random.uniform(low=-0.001, high=0.001, size=(nb_d2v_features, nb_latent_f + 1))  # side info weight matrix
        self.ada_cache_weights = np.zeros_like(self.weights)
        self.reg_lambda = reg_lambda
        self.cosine_lambda = cosine_lambda

    def step(self, input_vec, user_or_item_id, lr):
        label_vec = self.vector_dict[user_or_item_id]

        bias = np.array([1]).reshape((-1, 1))
        biased_input_vec = np.hstack([bias, input_vec])

        label_predict = np.dot(self.weights, biased_input_vec.T).reshape((-1,))  # vector
        error = label_vec - label_predict
        loss = np.mean(np.square(error))
        loss += self.reg_lambda * np.sqrt(np.sum(np.square(self.weights)))
        #loss += self.cosine_lambda * (1.0 - (np.sum(label_vec * label_predict) /
        #                                     (np.sqrt(np.sum(np.square(label_vec))) * np.sqrt(np.sum(np.square(label_predict))))))

        delta_weights = np.dot(error.reshape((-1, 1)), biased_input_vec.reshape((1, -1))) - self.reg_lambda * self.weights
        delta_input_vec = np.dot(self.weights[:, 1:].T, error)  # without bias

        self.ada_cache_weights += delta_weights ** 2

        # update parameters
        self.weights += lr * delta_weights / (np.sqrt(self.ada_cache_weights) + 1e-6)

        return loss, delta_input_vec

    def predict(self, input_vec):
        bias = np.array([1]).reshape((-1, 1))
        biased_input_vec = np.hstack([bias, input_vec])

        label_predict = np.dot(self.weights, biased_input_vec.T).reshape((-1,))  # vector
        return label_predict


def create_si_item_model(config, ratings):
    if 'si_item_d2v_model' in config:
        d2v_model = Doc2Vec.load(config['si_item_d2v_model'])
        feature_vec_dict = MovieIdToDocVec(d2v_model.docvecs, ratings)
    elif 'si_item_vector_dict' in config:
        feature_vec_dict = dd.io.load(config['si_item_vector_dict'])

    si_item_nn = list(config['si_item_nn_hidden'])
    si_item_nn.insert(0, config['nb_latent_f'])
    si_item_nn.append(int(feature_vec_dict[config['si_item_valid_id']].shape[0]))
    config['si_item_nn'] = si_item_nn
    #si_item_model = ATSideInfoModel(config['si_item_nn'], config['si_item_reg_lambda'], feature_vec_dict)
    si_item_model = NNSideInfoModel(config['si_item_nn'],
                                    config['si_item_reg_lambda'],
                                    config['si_item_cosine_lambda'],
                                    feature_vec_dict)
    return si_item_model, config


def create_si_user_model(config, ratings):
    if 'si_user_d2v_model' in config:
        d2v_model = Doc2Vec.load(config['si_user_d2v_model'])
        feature_vec_dict = UserIdToDocVec(d2v_model.docvecs, ratings)
    elif 'si_user_vector_dict' in config:
        feature_vec_dict = dd.io.load(config['si_user_vector_dict'])

    si_user_nn = list(config['si_user_nn_hidden'])
    si_user_nn.insert(0, config['nb_latent_f'])
    si_user_nn.append(int(feature_vec_dict[config['si_user_valid_id']].shape[0]))
    config['si_user_nn'] = si_user_nn
    si_user_model = NNSideInfoModel(config['si_user_nn'],
                                    config['si_user_reg_lambda'],
                                    config['si_user_cosine_lambda'],
                                    feature_vec_dict)
    return si_user_model, config
