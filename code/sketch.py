import numpy
import theano
import theano.tensor as T

numpy_rng = numpy.random.RandomState(1234)

filter_shape = [2, 3, 4]

numpy_V = numpy.asarray(numpy.ones(filter_shape), dtype=theano.config.floatX)

# V is a [layers =2, height=3, width=4] tensor
V = theano.shared(value=numpy_V, name='V', borrow=True)

# c is a [layer = 2] tensor that I want to apply to each pixel of the HxW layer of V
c = numpy.asarray([2, 3])

prod = V * c

output = theano.function([], prod)


