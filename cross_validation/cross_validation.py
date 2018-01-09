def k_fold_cross_validation(dataset, algorithm, loss_func, k):
    """
    Performs k-fold cross validation on a given dataset for a given algorithm
    with some loss function (scoring function).

    Parameters
    ----------
    first: dataset
        a list of objects that make up the dataset
    second: algorithm
        a function that takes a (sub)list of objects that make up the dataset
        as its sole argument and returns a function that takes a single dataset
        object and maps it to a scalar
    third: loss_func
        a function that returns a scalar given a learned algorithm followed
        by a dataset object as input.
    fourth: k
        the number of folds to make on the data

    Returns
    -------
    e
        a list that gives the error/loss measure for each object in the dataset
        whose mean is the estimated generalization error
    """
    if len(dataset) < k:
        raise ValueError('dataset must be able to be partitioned into k folds')
    e = []
    for i in xrange(0, k):
        (training_set, test_set) = __get_training_and_test_set(dataset, k, i)
        learned_algorithm = algorithm(training_set)
        for test in test_set:
            e.append(loss_func(learned_algorithm, test))
    return e

def __get_training_and_test_set(dataset, k, set_to_exclude):
    """
    A tuple where the first is the training set (made up of the union of all
    k chunks of the dataset subtracted by the chunk corresponding to the set
    to exclude ... and the second is the removed chunked

    Parameters
    ----------
    first: dataset
        a list of objects that makes up the dataset
    second: k
        the number of chunks to partition the dataset into
    third: set_to_exclude
        the index of the chunk to exclude (indexed at 0)

    Returns
    -------
        A tuple where the first is the training set (made up of the union of all
        k chunks of the dataset subtracted by the chunk corresponding to the set
        to exclude ... and the second is the removed chunked
    """
    if (set_to_exclude > k):
        raise ValueError('cannot exclude an index larger than k')
    training_set = []
    test_set = []
    chunks = __chunks(dataset, k)
    for i in xrange(0, k):
        if (i == set_to_exclude):
            test_set += chunks.next()
        else:
            training_set += chunks.next()
    return (training_set, test_set)

def __chunks(list, k):
    """
    Generator for successive chunks of a list

    Parameters
    ----------
    first: list
        the list to break into chunks
    second: k
        the number of chunks to break the list into

    Returns
    -------
        A generator for successive chunks of the list
    """
    for i in range(0, len(list), k):
        yield list[i:(i + k)]

if __name__ == "__main__":
    print k_fold_cross_validation([1,2,3,4,5,6,7,8,9], lambda d: lambda y: y-1, lambda l, x: l(x) , 3)
