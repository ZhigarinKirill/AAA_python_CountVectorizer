import unittest
from feature_extraction import CountVectorizer as MyCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class TestCountVectorizer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        print(f'{self._testMethodName} завершился')

    def test1(self):
        corpus = [
            'Crock Pot Pasta Never boil pasta again',
            'Pasta Pomodoro Fresh ingredients Parmesan to taste'
        ]
        my_vectorizer = MyCountVectorizer()
        my_count_matrix = my_vectorizer.fit_transform(corpus)

        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(corpus).toarray()

        self.assertListEqual(my_vectorizer.get_feature_names(),
                             list(vectorizer.get_feature_names_out()))
        self.assertTrue((my_count_matrix == count_matrix).all())

    def test2(self):
        corpus = [
            ' Crock Pot Pasta.   Never boil pasta again!!!',
            'Pasta Pomodoro;  Fresh ingredients; Parmesan to taste '
        ]
        my_vectorizer = MyCountVectorizer()
        my_count_matrix = my_vectorizer.fit_transform(corpus)

        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(corpus).toarray()

        self.assertListEqual(my_vectorizer.get_feature_names(),
                             list(vectorizer.get_feature_names_out()))
        self.assertTrue((my_count_matrix == count_matrix).all())

    def test3(self):
        corpus = [
            'The four seasons of the year are beautiful and pleasant. Summer is the most colourful season.',
            'Autumn brings all kinds of fruits and vegetables.',
            'We may also enjoy some warm and pleasant days in September.',
            'Winter covers everything with glittering snow.'
        ]
        my_vectorizer = MyCountVectorizer()
        my_count_matrix = my_vectorizer.fit_transform(corpus)

        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(corpus).toarray()

        self.assertListEqual(my_vectorizer.get_feature_names(),
                             list(vectorizer.get_feature_names_out()))
        self.assertTrue((my_count_matrix == count_matrix).all())


if __name__ == '__main__':
    unittest.main()
