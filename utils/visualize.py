import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _convert_to_df(word_id_map, embeddings_2d):
    df_data = []
    for word in word_id_map:
        df_data.append([word,
                        embeddings_2d[word_id_map[word], 0],
                        embeddings_2d[word_id_map[word], 1]])
    df = pd.DataFrame(df_data, columns=['word', 'x', 'y'])
    return df

def plot(word_id_map, embeddings_2d, x_bounds=None, y_bounds=None, plot_text=False):
    sns.set_context("poster")
    df = _convert_to_df(word_id_map, embeddings_2d)

    if x_bounds is not None and y_bounds is not None:
        slice = df[
            (x_bounds[0] <= df.x) & (df.x <= x_bounds[1]) &
            (y_bounds[0] <= df.x) & (df.y <= y_bounds[1])
        ]
    else:
        slice = df
    ax = slice.plot.scatter('x', 'y', s=10, figsize=(10, 10))
    if plot_text:
        for _, point in slice.iterrows():
            ax.text(point.x + 5e-3, point.y + 5e-3, point.word, fontsize=11)
