config = {
    'nltk_packages': ['punkt', 'stopwords'],
    'tokenizer_path': 'tokenizers/punkt/english.pickle',
    'data_root': './dataset/',
    'vocab_size': 10000,
    'model_dir': './word2vec',
    'epochs': 10,
    'batch_size': 32,
    'embedding_size': 128,
    'w2v_root': './generated_w2vs/',
    'num_neg_samples': 64
}