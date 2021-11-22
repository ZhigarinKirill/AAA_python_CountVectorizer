from feature_extraction import CountVectorizer, TfidfVectorizer, TfidfTransformer
from math import log


def tf_transform(count_matrix):
    tf_matrix = []
    for v in count_matrix:
        tf_matrix.append([freq/sum(v) for freq in v])
    return tf_matrix


def idf_transform(count_matrix):
    transposed_count_matrix = list(map(list, zip(*count_matrix)))
    idfs = [log((len(count_matrix) + 1)/(sum(1 for freq in word_freq if freq) +
                                         1)) + 1 for word_freq in transposed_count_matrix]
    return idfs


if __name__ == '__main__':
    # 1
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
    print()
    # 2
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]
    tf_matrix = tf_transform(count_matrix)
    print(tf_matrix)
    print()
    # 3
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]
    idf_matrix = idf_transform(count_matrix)
    print(idf_matrix)
    print()
    # 4
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)
    print(tfidf_matrix)
    print()
    # 5
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
    print()
