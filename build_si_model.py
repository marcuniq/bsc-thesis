import theano
import theano.tensor as T
import numpy as np


def build_si_model(dim_Qi, dim_d2v, nb_examples, reg_lambda):
    Qi, y = T.fvector('Qi'), T.fvector('y')
    W1 = theano.shared(np.random.randn(dim_Qi, dim_d2v) / np.sqrt(dim_Qi))
    b1 = theano.shared(np.zeros(dim_d2v))
    y_predict = Qi.dot(W1) + b1

    loss_reg = 1./nb_examples * reg_lambda/2 * (T.sum(T.sqr(W1)))
    loss = T.sqr(y - y_predict).sum() + loss_reg

    dQi = theano.grad(loss, Qi)
    dW1 = theano.grad(loss, W1)
    db1 = theano.grad(loss, b1)

    lr = T.scalar('lr')

    #forward_prop = T.function([Qi], y_predict, allow_input_downcast=True)
    calc_loss = theano.function([Qi, y], loss, allow_input_downcast=True)
    get_qi_grad = theano.function([Qi, y], dQi, allow_input_downcast=True)
    gradient_step = theano.function([Qi, y, lr], updates=((W1, W1 - lr * dW1), (b1, b1 - lr * db1)), allow_input_downcast=True)

    return calc_loss, get_qi_grad, gradient_step
