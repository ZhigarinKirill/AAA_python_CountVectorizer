import typing
import re


class Vectorizer:
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

    def fit_transform(self, corpus: typing.List[str],) -> typing.List[typing.List[str]]:
        pass


class CountVectorizer(Vectorizer):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, corpus: typing.List[str]) -> typing.List[typing.List[str]]:
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
            # yield list(freq_words.values())
        return freq_docs

    def fit_transform(self, corpus: typing.List[str]) -> typing.List[typing.List[str]]:
        self.fit(corpus)
        return self.transform(corpus)
