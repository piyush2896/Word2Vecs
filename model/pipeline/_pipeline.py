import tensorflow as tf

def input_fn(inputs,
             labels=None,
             batch_size=32,
             buffer_size=2000,
             shuffle=True,
             repeat=None):
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    else:
        dataset = dataset.prefetch(buffer_size)

    dataset = dataset.batch(batch_size).repeat(repeat)

    if labels is not None:
        inputs, labels = dataset.make_one_shot_iterator().get_next()
        return {'word_ids': inputs}, labels
    inputs, = dataset.make_one_shot_iterator().get_next()
    return {'word_ids': inputs}
