import numpy as np

def pca(input, l):
    """
    Performs PCA on an input of m points in n dimensional space, yielding
    a tuple containing two function that can be used to encode a given point
    and decode a given point, respectively. The two functions take a given point
    as an argument. The dimensionality of the encoding is specified by l.

    Parameters
    ----------
    first: input
        a tuple of m points, of which each point is represnted by an
        n-dimensional numpy array
    second: l
        the dimensionality of the resulting representation

    Returns
    -------
    tuple
        a tuple containing two function that can be used to encode a given point
        and decode a given point, respectively. The two functions take a given
        point as an argument.
    """
    # Get the eigenvalues and eigenvectors
    x = np.row_stack(input)
    xtx = x.T.dot(x)
    evalues, evectors = np.linalg.eig(xtx)
    # Determine the largest eigenvectors
    # We do this by creating a list of tuples where we have corresponding
    # (eigenvalue, eigenvector) and then sorting according to the eigenvalues
    eigens = map(lambda x: (evalues[x], evectors[x]), range(0, len(evalues)))
    sorted_eigens = sorted(eigens)
    # We construct the encoding/decoding matrix D
    d = np.column_stack(tuple([evector for _, evector in sorted_eigens]))[:,:l]
    encoder = lambda x: d.T.dot(x)
    decoder = lambda c: d.dot(c)
    return (encoder, decoder)

if __name__ == "__main__":
    encoder, decoder = pca((np.array((3,5,4,7,11)), np.array((3,7,9,12,15)), np.array((3,4,10,3,9)), np.array((8,2,6,11,12))), 5)
    print decoder(encoder(np.array((3,7,9,12,15))))
