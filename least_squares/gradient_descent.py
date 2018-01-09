import numpy as np

def gradient_descent(A, b, l, e, x):
    """
    Performs an optimization on f(x)=0.5||Ax-b||^2_2 using the gradient
    descent method to find the value of x that minimizes.

    Parameters
    ----------
    first: A
        a numpy array that specifies the matrix A in the given equation
    second: b
        a numpy array that specifies the vector b in the given equation
    third: l
        the tolerance in the optimization
    fourth: e
        the step size for searching for the optimization
    fifth: x
        the starting value to try for x

    Returns
    -------
    x
        a numpy array that minimizes f(x)=0.5||Ax-b||^2_2
    """
    derivative = (A.T.dot(A).dot(x)) - (A.T.dot(b))
    if (np.linalg.norm(derivative, ord=2) > l):
        return gradient_descent(A, b, l, e, x - derivative.dot(e))
    else:
        return x

if __name__ == "__main__":
    print gradient_descent(np.array([[1, 9],[4, 5]]), np.array((3,7)), 0.1, 0.01, np.array((4,5)))
