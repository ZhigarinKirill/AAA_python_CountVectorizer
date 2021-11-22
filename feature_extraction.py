import typing
import re
from math import log


def transpose(matrix):
    return list(map(list, zip(*matrix)))


class CountVectorizer:
    def __init__(self) -> None:
        self._features = list()

    def _tokenize(self, text: str) -> typing.List[str]:
        return text.split(' ')

    def _preprocess(self, text: str) -> typing.List[str]:
        text = re.sub(r'[,.;:"\'?!]+', '', text)
        text = re.sub(r'\s\s+', ' ', text).lower().strip()
        tokens = self._tokenize(text)
        return tokens

    def fit(self, corpus: typing.List[str]) -> None:
        feature_set = set()
        for doc in corpus:
            preprocessed_doc = self._preprocess(doc)
            feature_set |= set(preprocessed_doc)
        self._features = sorted(feature_set)

    def get_feature_names(self) -> typing.List[str]:
        return self._features

    def _to_count_matrix(self, corpus: typing.List[str]) -> typing.List[typing.List[str]]:
        if not self._features:
            return []
        freq_docs = list()
        for doc in corpus:
            preprocessed_doc = self._preprocess(doc)
            freq_words = dict.fromkeys(self._features, 0)
            for token in preprocessed_doc:
                if token in freq_words:
                    freq_words[token] += 1
            freq_docs.append(list(freq_words.values()))
        return freq_docs

    def transform(self, corpus: typing.List[str]) -> typing.List[typing.List[str]]:
        return self._to_count_matrix(corpus)

    def fit_transform(self, corpus: typing.List[str]) -> typing.List[typing.List[str]]:
        self.fit(corpus)
        return self.transform(corpus)


class TfidfTransformer:
    def __init__(self):
        pass

    def _compute_tfs(self, count_matrix: typing.List[typing.List[int]]) -> typing.List[typing.List[float]]:
        tf_matrix = [[freq/sum(vec) for freq in vec] for vec in count_matrix]
        return tf_matrix

    def _compute_idfs(self, count_matrix: typing.List[typing.List[int]]) -> typing.List[float]:
        transposed_count_matrix = transpose(count_matrix)
        idfs = [log((len(count_matrix) + 1)/(sum(1 for freq in word_freq if freq) +
                                             1)) + 1 for word_freq in transposed_count_matrix]
        return idfs

    def fit_transform(self, count_matrix: typing.List[typing.List[int]]) -> typing.List[typing.List[float]]:
        tfs = self._compute_tfs(count_matrix)
        idfs = self._compute_idfs(count_matrix)
        return [[tf*idf for tf, idf in zip(doc_tfs, idfs)] for doc_tfs in tfs]


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self._tranformer = TfidfTransformer()

    def fit_transform(self, corpus: typing.List[str]) -> typing.List[typing.List[float]]:
        count_matrix = super().fit_transform(corpus)
        return self._tranformer.fit_transform(count_matrix)
