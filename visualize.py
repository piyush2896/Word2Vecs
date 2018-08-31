import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils import unpickle
from utils.visualize import plot
from utils import config

def plot_embeddings(word_id_map,
                    embeddings_2d,
                    x_bounds=None,
                    y_bounds=None,
                    plot_text=False):
    plot(word_id_map, embeddings_2d, x_bounds, y_bounds, plot_text)
    plt.show()

def run():
    print('* Reading files')
    if not os.path.isdir(config['w2v_root']):
        raise FileNotFoundError('Files directory not found')
    if not os.path.isfile(os.path.join(config['w2v_root'], 'word_ids.pickle')):
        raise FileNotFoundError('File word_ids.pickle not found')
    if not os.path.isfile(os.path.join(config['w2v_root'], 'embedding_matrix.npy')):
        raise FileNotFoundError('File embedding_matrix.npy not found')

    word_id_map = unpickle(os.path.join(config['w2v_root'], 'word_ids.pickle'))
    embeddings = np.load(os.path.join(config['w2v_root'], 'embedding_matrix.npy'))

    print('* Applying TSNE')
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    print('* Plotting All embeddings')
    plot_embeddings(word_id_map, embeddings_2d)
    print('* Plotting x-bounds (4.0, 4.2) and y-bounds (-0.5, -0.1)')
    plot_embeddings(word_id_map, embeddings_2d,
                    x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1),
                    plot_text=True)
    print('* Plotting x-bounds (14, 17) and y-bounds (4, 7.5)')
    plot_embeddings(word_id_map, embeddings_2d,
                    x_bounds=(14, 17), y_bounds=(4, 7.5), plot_text=True)

if __name__ == '__main__':
    run()
