import lasagne
import theano
import theano.tensor as T
import numpy as np


class NNSideInfoModel(object):
    """Neural Network"""
    def __init__(self, dim, reg_lambda):
        Qi = T.matrix('Qi', dtype=theano.config.floatX)
        f = T.vector('f', dtype=theano.config.floatX)

        network = lasagne.layers.InputLayer(shape=(None, dim[0]), input_var=Qi)
        for i in range(1, len(dim)-1):
            n_out = dim[i]
            network = lasagne.layers.DenseLayer(network, num_units=n_out, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network, num_units=dim[-1])
        f_predict = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(f, f_predict).sum()
        loss += reg_lambda * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

        dQi = theano.grad(loss, Qi)
        lr = T.scalar('lr', dtype=theano.config.floatX)

        # create parameter update expressions
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, lr)

        self.gradient_step = theano.function([Qi, f, lr], outputs=[loss, dQi], updates=updates, allow_input_downcast=True)


class ATSideInfoModel(object):
    """Affine Transformation"""
    def __init__(self, dim, reg_lambda):
        nb_latent_f = dim[0]
        nb_d2v_features = dim[-1]

        self.G = np.random.uniform(low=-0.001, high=0.001, size=(nb_d2v_features, nb_latent_f + 1))  # side info weight matrix
        self.ada_cache_G = np.zeros_like(self.G)
        self.reg_lambda = reg_lambda

    def gradient_step(self, Qi, feature, lr):
        bias = np.array([1]).reshape((-1, 1))
        biased_qi = np.hstack([bias, Qi])

        feature_predict = np.dot(self.G, biased_qi.T).reshape((-1,))  # vector
        feature_error = feature - feature_predict
        loss = np.sum(np.square(feature_error))

        deltaG = np.dot(feature_error.reshape((-1, 1)), biased_qi.reshape((1, -1))) - self.reg_lambda * self.G
        deltaQ_i = np.dot(self.G[:, 1:].T, feature_error)  # without bias

        self.ada_cache_G += deltaG ** 2

        # update parameters
        self.G += lr * deltaG / (np.sqrt(self.ada_cache_G) + 1e-6)

        return loss, deltaQ_i
