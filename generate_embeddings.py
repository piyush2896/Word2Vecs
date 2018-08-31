# installed or python modules
import tensorflow as tf
import numpy as np
import glob
import os
import logging

# defined modules
from utils import config, pickle_obj
from model.pipeline import preprocess
from model.pipeline import input_fn
from model import word2vec

def train_and_save():
    # logging to file
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.DEBUG)

    fhandler = logging.FileHandler('tensorflow.log')
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(fhandler)

    # Preprocess the files
    sgm_preprocessor = preprocess.SkipGramPreprocess(config['nltk_packages'],
                                                     config['tokenizer_path'],
                                                     config['data_root'],
                                                     config['vocab_size'])
    # Generate context target pairs
    context_target_pairs = preprocess.SkipGramContextTargetPair(
        sgm_preprocessor, seed=42).get_context_target_pairs(20)

    np.random.seed(42)
    np.random.shuffle(context_target_pairs)

    contexts = context_target_pairs[:, 0]
    targets = np.expand_dims(context_target_pairs[:, 1], 1)
    input_fn_ = lambda: input_fn(contexts, targets, batch_size=config['batch_size'])
    w2v = tf.estimator.Estimator(model_fn=word2vec,
                                 model_dir=config['model_dir'],
                                 params={
                                     'vocab_size': config['vocab_size'],
                                     'embedding_size': config['embedding_size'],
                                     'num_sampled': config['num_neg_samples']
                                 })

    steps = config['epochs'] * contexts.shape[0] // config['batch_size']
    print('* Starting to train')
    print('\t- Number of epochs: {0:,}'.format(config['epochs']))
    print('\t- Number of steps : {0:,}'.format(steps))
    w2v.train(input_fn=input_fn_, steps=steps)
    print('* End of training')
    print('\t- For training logs see tensorflow.log')

    print('* Collecting Embedding matrix')
    input_fn_ = lambda: input_fn(contexts[:10], targets[:10], repeat=1)
    embedding_matrix = next(w2v.predict(input_fn_))['embedding_matrix']

    # Save embeddings
    print('* Saving Embeddings')
    if not os.path.isdir(config['w2v_root']):
        os.makedirs(config['w2v_root'])
    pickle_obj(sgm_preprocessor.word_to_ids,
               os.path.join(config['w2v_root'], 'word_ids.pickle'))
    np.save(os.path.join(config['w2v_root'], 'embedding_matrix.npy'),
            embedding_matrix)

if __name__ == '__main__':
    train_and_save()
