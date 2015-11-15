import json
import random
import re
import time
import nltk

from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__author__ = 'tpc 2015'

word_regexp = re.compile(u"(?u)#?\w+"
                         u"|:\)+"
                         u"|;\)+"
                         u"|:\-\)+"
                         u"|;\-\)+"
                         u"|\(\(+"
                         u"|\)\)+"
                         u"|!+"
                         u"|\?+"
                         u"|\+[0-9]+"
                         u"|\++"
                         u"|\.")


rustem = nltk.stem.snowball.RussianStemmer()
stop_words_regex = None


def my_tokenizer(text):
    tokens = word_regexp.findall(text.lower())
    filtered_tokens = []

    for token in tokens:
        ch = token[0]
        if ch == ':' or ch == ';':
            token = ':)'
        elif ch == '(':
            token = '('
        elif ch == ')':
            token = ')'
        elif ch == '?':
            token = '?'
        elif ch == '!':
            token = '!'
        elif ch == '+':
            token = '+'
        elif ch == '#':
            token = '#'
        elif ch.isdigit() or stop_words_regex.match(token):
            continue
        filtered_tokens.append(token)
    return filtered_tokens


def my_analyzer(text):
    ngram_range = 1
    words = my_tokenizer(text)
    ngram_words = []

    for w in words:
        # w = rustem.stem(w)

        if len(w) == 0:
            continue

        if w == '.':
            ngram_words = []
            continue

        ngram = ""
        ngram_words.insert(0, w)
        for p in ngram_words:
            ngram = p + ' ' + ngram
            yield ngram[:-1]
        if len(ngram_words) > ngram_range:
            ngram_words.pop()

        if w[0] == ':' or w[0] == '?' or w[0] == '!' or w[0] == '+':
            ngram_words = []


class CustomFeatures(TransformerMixin):
    def fit(self, X, y=None, **params):
        return self

    def transform(self, instances, y=None, **fit_params):
        features = []
        for instance in instances:
            features.append(self._text_features(instance))
        return features

    @staticmethod
    def _text_features(instance):
        features = []

        max_len = 0
        ends_with_smile = 0
        ends_with_bracket = 0
        for text in instance:
            tokens = my_tokenizer(text)
            if len(tokens) > 0 and tokens[-1] == ':)':
                ends_with_smile += 1
            if len(tokens) > 0 and tokens[-1] == ')':
                ends_with_bracket += 1

            if len(text) > max_len:
                max_len = len(text)

        features.append(float(max_len))
        features.append(ends_with_smile / len(instance))
        features.append(ends_with_bracket / len(instance))
        return features


class ConcatenateTransformer(TransformerMixin):
    def fit(self, X, y=None, **params):
        return self

    def transform(self, instances, y=None, **fit_params):
        new_instances = []
        for instance in instances:
            concatenated = ''
            for text in instance:
                concatenated += ' ' + text
            new_instances.append(concatenated)
        return new_instances


class AgeDetector:
    def __init__(self):
        with open('stop_words.txt', mode='r', encoding='utf-8') as f:
            self.stop_words = f.read().splitlines()
        self.stop_words_regex = self.create_regex(self.stop_words)

        global stop_words_regex
        stop_words_regex = self.stop_words_regex

        self.text_clf = None

    @staticmethod
    def create_regex(expression_list):
        regex_str = '^('
        for exp in expression_list:
            regex_str += exp + '|'
        regex_str = regex_str[:-1] + ')$'
        regex = re.compile(regex_str)
        return regex

    @staticmethod
    def _make_clf():
        return Pipeline([
            ('vect', FeatureUnion([
                ('tfidf', Pipeline([
                    ('concat', ConcatenateTransformer()),
                    ('tfidf', TfidfVectorizer(analyzer=my_analyzer))
                ])),
                ('cust', Pipeline([
                    ('cust', CustomFeatures()),
                    ('scaler', StandardScaler())
                ]))
            ])),
            ('clf', SGDClassifier(alpha=5e-04,
                                  penalty='l2',
                                  loss='hinge',
                                  n_iter=50))
        ])

    def train(self, instances, labels):
        global stop_words_regex
        stop_words_regex = self.stop_words_regex

        text_clf = self._make_clf()
        self.text_clf = text_clf.fit(instances, labels)

    def classify(self, instances):
        return self.text_clf.predict(instances)

    def _cross_validate(self, instances, labels):
        text_clf = self._make_clf()
        score = cross_validation.cross_val_score(text_clf,
                                                 instances,
                                                 labels,
                                                 cv=5,
                                                 scoring='accuracy',
                                                 # n_jobs=-1,
                                                 verbose=5)
        print(score)

    def test_tokenizer(self, instances, labels):
        for i in range(50):
        # for i in range(len(instances)):
            instance = instances[i]
            label = labels[i]
            try:
                print(instance)
                for analyzed in my_analyzer(instance):
                    print(analyzed)
                    pass
            except:
                pass
        exit()

    def _test_split(self, instances, labels):
        start_time = time.time()

        instances, instances_test, labels, labels_test \
            = cross_validation.train_test_split(instances, labels, test_size=0.2, random_state=2)

        print('Starting train')
        self.train(instances, labels)
        print('Training done')

        print(accuracy_score(labels_test, self.classify(instances_test)))
        print("--- %.1f minutes ---" % ((time.time() - start_time) / 60))

    def test(self):
        json_file_labels = open('Train.lab.json', encoding='utf-8', errors='replace')
        json_file_instances = open('Train.txt.json', encoding='utf-8', errors='replace')

        labels = json.load(json_file_labels)
        instances = json.load(json_file_instances)

        for i in range(20):
            j = random.randint(0, len(instances))
            try:
                print(instances[j])
                print(labels[j])
            except:
                pass

        # self.test_tokenizer(instances, labels)
        # self._cross_validate(instances, labels)
        # self._grid_search(instances, labels)
        # self.train(instances, labels)
        self._test_split(instances, labels)

if __name__ == '__main__':
    d = AgeDetector()
    d.test()
