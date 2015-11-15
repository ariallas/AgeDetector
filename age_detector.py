import json
import re
import time
import nltk

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

__author__ = 'tpc 2015'

word_regexp = re.compile(u"(?u)\w+"
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
    def _concatenate_instances(instances):
        new_instances = []
        for instance in instances:
            concatenated = ""
            for text in instance:
                concatenated += ' ' + text
            new_instances.append(concatenated)
        return new_instances

    @staticmethod
    def _unfold_instances(instances, labels):
        new_instances = []
        new_labels = []
        for i in range(len(instances)):
            for text in instances[i]:
                new_instances.append(text)
                new_labels.append(labels[i])
        return new_instances, new_labels

    @staticmethod
    def _make_clf():
        return Pipeline([
            ('vect', TfidfVectorizer(analyzer=my_analyzer)),
            ('clf', SGDClassifier(alpha=3e-05,
                                  penalty='l2',
                                  loss='hinge',
                                  n_iter=50))
        ])
    # 3-5 468


    def train(self, instances, labels):
        instances = self._concatenate_instances(instances)

        global stop_words_regex
        stop_words_regex = self.stop_words_regex

        text_clf = self._make_clf()
        self.text_clf = text_clf.fit(instances, labels)

    def classify(self, instances):
        instances = self._concatenate_instances(instances)
        return self.text_clf.predict(instances)

    def _cross_validate(self, instances, labels):
        instances, labels = self._unfold_instances(instances, labels)

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
        instances, labels = self._unfold_instances(instances, labels)

        for i in range(10000):
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

        # self.test_tokenizer(instances, labels)
        # self._cross_validate(instances, labels)
        # self._grid_search(instances, labels)
        # self.train(instances, labels)
        self._test_split(instances, labels)

if __name__ == '__main__':
    d = AgeDetector()
    d.test()
