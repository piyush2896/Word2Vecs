from collections import defaultdict
import nltk
import glob
import os
import codecs
import re
import random
import numpy as np

class SkipGramPreprocess:
    def __init__(self, nltk_packages, tokenizer_path, files_root, vocab_size):
        for package in nltk_packages:
            nltk.download(package)
        self.tokenizer = nltk.data.load(tokenizer_path)
        self.vocab_size = vocab_size
        self.files = glob.glob(os.path.join(files_root, '*'))
        self._read_given_corpus()
        self.raw_sentences = self.tokenizer.tokenize(self.raw_corpus)
        self._generate_sentences()
        self._generate_word_ids()
        print('* Generated {0:,} word ids'.format(len(self.word_to_ids)))

    def _get_word_counts(self):
        word_counts = defaultdict(lambda: 0)
        for sentence in self.sentences:
            for word in sentence:
                word_counts[word.lower()] += 1
        return word_counts

    def _generate_word_ids(self):
        word_counts = self._get_word_counts()
        word_counts_tuple = list(zip(word_counts.keys(), word_counts.values()))
        top_k_word_counts = sorted(word_counts_tuple,
                                   key=lambda x: x[1],
                                   reverse=True)[:self.vocab_size]
        top_k_words = [word for word, _ in top_k_word_counts]
        self.word_to_ids = dict(zip(top_k_words,
                                list(range(self.vocab_size))))
        self.id_to_words = dict(zip(list(range(self.vocab_size)),
                                    top_k_words))

    def _read_given_corpus(self):
        self.raw_corpus = u""
        for i, filename in enumerate(self.files):
            with codecs.open(filename, "r", "utf-8") as f:
                print('* Reading book {} from path {}'.format(i, filename))
                self.raw_corpus += f.read()
        print('* Corpus length = {0:,} charachters'.format(len(self.raw_corpus)))

    def _generate_sentences(self):
        def sentence_to_wlist(raw_sentence):
            clean = re.sub("[^a-zA-Z]", " ", raw_sentence)
            return clean.split()
        self.sentences = []
        for raw_sentence in self.raw_sentences:
            self.sentences.append(sentence_to_wlist(raw_sentence))
        print('* Number of Tokens found {0:,}'.format(sum([len(sentence) for sentence in self.sentences])))

    def get_padded_sentences(self, min_threshold=3, max_seq_len=100):
        sentences = []
        for sentence in self.sentences:
            if len(sentence) < min_threshold or len(sentence) > max_seq_len:
                continue
            sentence_ids = []

            for word in sentence:
                if word.lower() in self.word_to_ids:
                    sentence_ids.append(self.word_to_ids[word.lower()])
                else:
                    sentence_ids.append(self.vocab_size)

            sentences.append(sentence_ids)
        return sentences

class SkipGramContextTargetPair:
    def __init__(self, sgm_preprocessor, window_size=3, seed=None):
        self.sgm_preprocessor = sgm_preprocessor
        self.window_size = window_size
        self.seed = seed

    def _generate_pairs(self):
        sentences = self.sgm_preprocessor.get_padded_sentences()
        context_target_dict = defaultdict(lambda: [])
        for sentence in sentences:
            for word_index, word in enumerate(sentence):
                if word == self.sgm_preprocessor.vocab_size:
                    continue
                if word_index < self.window_size:
                    context_target_dict[word].extend(
                        sentence[:word_index] + sentence[word_index + 1:word_index + 1 + self.window_size])
                elif len(sentence) - word_index > self.window_size:
                    context_target_dict[word].extend(
                        sentence[word_index - self.window_size:word_index] + sentence[word_index + 1:word_index + 1 + self.window_size])
                else:
                    context_target_dict[word].extend(
                        sentence[word_index - self.window_size:word_index] + sentence[word_index + 1:])
        return context_target_dict

    def get_context_target_pairs(self, targets_per_context):
        context_target_dict = self._generate_pairs()
        context_target_pairs = []
        if self.seed is not None:
            random.seed(self.seed)
        for context in context_target_dict:
            if len(context_target_dict[context]) < targets_per_context:
                for word in context_target_dict[context]:
                    context_target_pairs.append([context, word])
            else:
                for _ in range(targets_per_context):
                    context_target_pairs.append([context, random.choice(context_target_dict[context])])
        print('* Generated {0:,} context target pairs'.format(len(context_target_pairs)))
        return np.array(context_target_pairs)
