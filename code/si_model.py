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
        Qi = T.matrix('Qi', dtype=theano.config.floatX)
        f = T.vector('f', dtype=theano.config.floatX)

        W1 = theano.shared(np.random.randn(dim[0], dim[1]) / np.sqrt(dim[0]))
        b1 = theano.shared(np.zeros(dim[1]))
        f_predict = Qi.dot(W1) + b1

        loss = T.sqr(f - f_predict).sum()
        loss += reg_lambda * (T.sum(T.sqr(W1)))

        dQi = theano.grad(loss, Qi)
        dW1 = theano.grad(loss, W1)
        db1 = theano.grad(loss, b1)

        lr = T.scalar('lr', dtype=theano.config.floatX)

        self.gradient_step = theano.function([Qi, f, lr],
                                             outputs=[loss, dQi],
                                             updates=((W1, W1 - lr * dW1), (b1, b1 - lr * db1)),
                                             allow_input_downcast=True)
