class Variable:

    def __init__(self, array):
    """
     A variable in a computational graph. For our purposes this is restricted
     to being an array (note that we can represent a scalar using arrays with a
     1x1 array, so it's really scalars too).

    Attributes
    ----------
    array : numpy.array
        The value that this variable stores.

    Reference: Deep Learning 6.5.1
    """
        self.array = array

class Operation:

    def __init__(self, op, input_arrays, output_array):
    """
     An operation in a computational graph. For our purposes this is restricted
     to being an operation from and to arrays (note that we can represent a
     scalar using arrays with a 1x1 array, so it's really scalars too).

    Attributes
    ----------
    op :  function(numpy.array, numpy.array, ...) -> numpy.array
        Function that takes the input_arrays as arguments in the given order
        and returns an array (note that we can represent a scalar using arrays
        with a 1x1 array, so it's really scalars too).
    input_arrays : list[numpy_array]
        A list of numpy arrays that can be passed into op as arguments in the
        given order and allow for proper execution of op
    output_array : numpy_array
        An array that is set after calling exewcute...

    Reference: Deep Learning 6.5.1
    """
        self.op = op
        self.input_arrays = input_arrays
        self.output_array = output_array
