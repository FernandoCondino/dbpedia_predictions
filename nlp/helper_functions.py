import re
import pandas as pd
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from dbpedia_common import TARGET_HELD_OUT_CSV, TARGET_DEV_CSV
from collections import deque
from scipy.sparse import vstack as sparse_vstack
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

SHORT_ABSTRACTS_PICKLE = 'cache/short_abstracts.p'
RANDOM_STATE = 23

BIN_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]
BIN_LABELS_TEXT = ['[1,000-2,500)', '[2,500-5,000)', '[5,000-10,000)', '[10,000-50,000)', '[50,000-100,000)',
                   '[100,000-500,000]', '[500,000-1,000,000)', '[1,000,000-+inf)']
BINS = [0, 2499, 4999, 9999, 49999, 99999, 499_999, 999_999, np.inf]


def discretize_target(df, bin_labels, bins):
    return pd.cut(df["target"], bins=bins, labels=bin_labels)


def get_short_descriptions():
    input_prefix = '../' if os.getcwd().endswith('nlp') else ''  # TODO: This is because it's difficult to know where is the root path with notebooks
    if os.path.isfile(SHORT_ABSTRACTS_PICKLE):
        abstracts = pickle.load(open(SHORT_ABSTRACTS_PICKLE, 'rb'))
    else:
        # Searching abstracts (descriptions) of all subjects we have in dev and held out sets
        target_df = pd.read_csv(input_prefix + TARGET_DEV_CSV).append(pd.read_csv(input_prefix + TARGET_HELD_OUT_CSV))
        filename = input_prefix + 'input/raw_data/short-abstracts_lang=en.ttl'
        list_of_subjects = set(target_df.subject.tolist())
        abstracts = _parse_short_abstracts(filename, list_of_subjects)
        pickle.dump(abstracts, open(SHORT_ABSTRACTS_PICKLE, 'wb'))
    return abstracts


def evaluate(pred, test_y, plot=True):
    if not plot:
        print('***Train***')
    print(f'Quadratic Kappa Score: {round(cohen_kappa_score(test_y, pred, weights="quadratic"), 2)}')
    print(f'F1 score Micro: {round(f1_score(test_y, pred, average="micro"), 2)}')
    print(f'F1 score Macro: {round(f1_score(test_y, pred, average="macro"), 2)}')

    if plot:
        cm = confusion_matrix(test_y, pred, labels=BIN_LABELS)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cm, index=[i for i in BIN_LABELS_TEXT], columns=[i for i in BIN_LABELS_TEXT])
        plt.figure(figsize=(15, 8))
        sns.set(font_scale=1.3)
        sns.heatmap(df_cm, annot=True, cmap='OrRd')


def _parse_short_abstracts(filename, list_of_subjects):
    abstracts = {}
    db_pedia_regex = re.compile('(<.+?>)\s+(<.+?>)\s+(.+?) \.')
    with open(filename, 'r') as file_reader:
        for line in file_reader:
            matcher = db_pedia_regex.match(line)
            if matcher:
                subject, verb, obj = matcher.groups()
                if subject in list_of_subjects:
                    abstracts[subject] = obj
            else:
                print('********** did not match!:', line)
    return abstracts


PUNCTUATION = ''.join(set(string.punctuation) - set('@'))


def words_analyser(doc):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stopwords_english = stopwords.words('english')
    tokens = deque()

    doc = doc.translate(str.maketrans('', '', PUNCTUATION)).lower()
    if doc.endswith('@en'):
        doc = doc[:-3]
    else:
        doc = doc + ' NOT_EN_LANG'
    for token in nltk.word_tokenize(doc):
        try:
            token = _to_int_token(int(token))
            if not token:
                continue
        except Exception:
            pass  # It isn't a number, so we continue
        if token not in stopwords_english and len(token) > 1:
            # print(token) # For debugging
            stemmed_token = stemmer.stem(token)
            tokens.append(stemmed_token)
            yield stemmed_token
    for n_gram in nltk.ngrams(list(tokens), 2):
        yield ' '.join(n_gram)


def _to_int_token(token_int):
    if token_int < 1000:
        result = None
    elif token_int > 1400 and token_int < 2019:
        result = 'POSSIBLE_DATE'
    elif token_int < 2500:
        result = '<2,500'
    elif token_int < 5000:
        result = '<5,000'
    elif token_int < 10000:
        result = '<10,000'
    elif token_int < 50000:
        result = '<50,000'
    elif token_int < 100000:
        result = '<100,000'
    elif token_int < 500000:
        result = '<500,000'
    elif token_int < 1000000:
        result = '<1,000,000'
    else:
        result = '+1,000,000'
    return result


def cross_validate(model, train_df, upsample=False, log=True):
    model_name = str(model)
    strat_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    scores = {'kappa': [], 'f1_micro': [], 'f1_macro': []}
    for train_index, test_index in strat_split.split(train_df, train_df['target']):
        X_train, X_test = train_df.loc[train_index].drop(columns='target'), train_df.loc[test_index].drop(
            columns='target')
        y_train, y_test = train_df.loc[train_index, 'target'], train_df.loc[test_index, 'target']

        stem_vectorizer = CountVectorizer(analyzer=words_analyser, min_df=3)
        tfidf_transf = TfidfTransformer()

        word_count_vector = stem_vectorizer.fit_transform(X_train.description)
        tfidf_vector = tfidf_transf.fit_transform(word_count_vector)

        if upsample:
            X_train_sampled, y_train_sampled = get_upsampling(train_df.loc[train_index],
                                                              count_threshold=8000,
                                                              tfidf_vector=tfidf_vector)
            model = model.fit(X_train_sampled, y_train_sampled)
        else:
            model = model.fit(tfidf_vector, y_train)

        x_test_transformed = tfidf_transf.transform(stem_vectorizer.transform(X_test.description))
        pred = model.predict(x_test_transformed)

        scores['kappa'].append(cohen_kappa_score(y_test, pred, weights="quadratic"))
        scores['f1_micro'].append(f1_score(y_test, pred, average="micro"))
        scores['f1_macro'].append(f1_score(y_test, pred, average="macro"))
    if log:
        print(model_name)
        print(f'Quadratic Kappa Score CV 5: {round(np.mean(scores["kappa"]), 2)}')
        print(f'F1 score Micro CV 5: {round(np.mean(scores["f1_micro"]), 2)}')
        print(f'F1 Score Macro CV 5: {round(np.mean(scores["f1_macro"]), 2)}')
        print('***************************************')
    return scores


def get_upsampling(train_df, count_threshold, tfidf_vector):
    train_df_copy = train_df.copy().reset_index(drop=True)  # needed for tfidf_vector.getrow(index)
    dfs_by_target = []
    max_df_size = 0
    sizes = train_df_copy.target.value_counts().to_dict()
    for target in train_df_copy.target.unique():
        df = train_df_copy[train_df_copy['target'] == target]
        if sizes[target] >= count_threshold:
            dfs_by_target.append(df)
        else:
            dfs_by_target.append(df.sample(max(sizes.values()), replace=True))
    X_train = sparse_vstack([tfidf_vector.getrow(i) for df in dfs_by_target for i in df.index])
    y_train = pd.concat(dfs_by_target, axis=0).target
    return X_train, y_train


def nlp_predict(X_train, y_train, X_test, upsampling_threshold):
    stem_vectorizer = CountVectorizer(analyzer=words_analyser, min_df=3)
    tfidf_transf = TfidfTransformer()

    word_count_vector = stem_vectorizer.fit_transform(X_train.description)
    tfidf_vector = tfidf_transf.fit_transform(word_count_vector)

    X_train_sampled, y_train_sampled = get_upsampling(pd.concat([X_train, y_train], axis=1),
                                                      count_threshold=upsampling_threshold,
                                                      tfidf_vector=tfidf_vector)
    x_test_transformed = tfidf_transf.transform(stem_vectorizer.transform(X_test.description))
    model = get_nlp_model()
    model = model.fit(X_train_sampled, y_train_sampled)

    return model.predict(x_test_transformed)


def get_nlp_model():
    return StackingClassifier(estimators=[
                               ('modified_huber_SGD', SGDClassifier(loss="modified_huber", alpha=0.002, penalty="l2", max_iter=10000)),
                               ('LogisticRegression', LogisticRegression(max_iter=10000, C=0.1, class_weight='balanced'))
                           ],
                           final_estimator=LogisticRegression(max_iter=10000, C=0.1, class_weight='balanced'))
