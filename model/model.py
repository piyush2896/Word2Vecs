import tensorflow as tf

def word2vec(features, labels, mode, params):
    word_embeddings = tf.get_variable("word_embeddings",
                                      [params['vocab_size'],
                                      params['embedding_size']])
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings,
                                               features['word_ids'])
    nce_weights = tf.get_variable("nce_weights",
                                  [params['vocab_size'],
                                   params['embedding_size']])
    nce_biases = tf.Variable(tf.zeros([params['vocab_size']], name='nce_biases'))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'embedding_matrix': tf.expand_dims(tf.convert_to_tensor(word_embeddings), 0)
        }
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=labels,
                           inputs=embedded_word_ids,
                           num_sampled=params['num_sampled'],
                           num_classes=params['vocab_size']))
        optimizer = tf.train.GradientDescentOptimizer(1.0)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return spec
