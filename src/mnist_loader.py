
""" 
mnist_loader
"""


import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

    def load_data_wrapper():
        """ Return a tuple containg `` (trining_data, validation_data, test_data)``. Based on ``load_data``, but the format is more convenient for use in our implementation of neural networks. In particular, ``training_data`` is a list containing 50,000 2-tuples ``(x,y)``. ``x`` is a 784-dimensioal numpy.ndarray containg the input image. ``y`` is the corresponding classification, i.e, the digit values (integers) corresponding to ``x``. 
        Obviously, this means we're using slightly different formats for the training data and the validation / test data. These formats turn out to be the most convenient for use in our neural network code """

        tr_d, va_d, te_d = load_data()

        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [vectorized_result(y) for y in tr_d[1]]
        training_data = zip(training_inputs, training_results)

        validaition_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validaition_inputs, va_d[1])

        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])

        return (training_data, validation_data, test_data)

    def vectorized_result(j):
        """ Return a 10-dimesional unit vector with a 1.0 in the jth position and zeroes elsewhere. This is used to convert a digit (0...9) into a corresponding desired output from the neural netowrk """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
