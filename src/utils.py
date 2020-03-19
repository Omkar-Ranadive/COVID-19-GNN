import pickle


def save_pickle(file, filename):
    """

    Args:
        file: file to be saved
        filename (str): path of the file to be saved

    Returns:

    """
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def load_pickle(filename):
    """

    Args:
        filename (str): name of the pickle file to be loaded

    Returns:
        file:

    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file
