from .config import config
import pickle

def pickle_obj(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def unpickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)