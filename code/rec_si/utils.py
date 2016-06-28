#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import unicodedata
# for timing
from contextlib import contextmanager
from timeit import default_timer

import progressbar
import time

def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.
    Return input string with accents removed, as unicode.
    deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'
    """
    if not isinstance(text, unicode):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = unicode('').join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def movie_to_imdb(ratings):
    result = {}

    def create_dict(df):
        result[df['movie_id'].unique()[0]] = df['imdb_id'].unique()[0]

    ratings.groupby('movie_id').apply(lambda df: create_dict(df))

    return result


def binarize_ratings(df, pos=1, neg=0, threshold=1, copy=False):
    copy = df.copy() if copy else df
    liked = copy['rating'] >= threshold
    disliked = copy['rating'] < threshold
    copy.loc[liked, 'rating'] = pos
    copy.loc[disliked, 'rating'] = neg
    return copy


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {k: v for d in dict_args for k, v in d.iteritems()}
    return result


def easy_parallize(f, params, p=8, nb_jobs=None):
    from multiprocessing import Pool, Manager
    pool = Pool(processes=p)
    m = Manager()
    q = m.Queue()

    nb_jobs = len(params) if nb_jobs is None else nb_jobs
    bar = progressbar.ProgressBar(max_value=nb_jobs)

    args = [(i, q) for i in params]
    results = pool.map_async(f, args)

    print "done dispatching..."
    bar.start()

    while not results.ready():
        complete_count = q.qsize()
        bar.update(complete_count)
        time.sleep(.5)

    bar.finish()

    pool.close()
    pool.join()

    return results.get()


class TransformedDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


class MovieIdToDocVec(TransformedDict):
    """Maps movie id to document vector"""
    def __init__(self, docvecs, ratings):
        self.store = docvecs
        self.movie_to_imdb = movie_to_imdb(ratings)

    def __keytransform__(self, key):
        return '{}.txt'.format(self.movie_to_imdb[key])


class UserIdToDocVec(TransformedDict):
    """Maps user id to document vector"""
    def __init__(self, d2v_model, ratings):
        self.d2v_model = d2v_model
        self.store = d2v_model.docvecs
        self.movie_to_imdb = movie_to_imdb(ratings)

    def __keytransform__(self, key):
        return '{}.txt'.format(self.movie_to_imdb[key])


def create_lookup_tables(ratings, user_key='user_id', item_key='movie_id'):
    users_unique = ratings[user_key].unique()
    nb_users = len(users_unique)
    users = dict(zip(users_unique, range(nb_users)))  # lookup table for user_id to zero indexed number

    movies_unique = ratings[item_key].unique()
    nb_movies = len(movies_unique)
    items = dict(zip(movies_unique, range(nb_movies)))  # lookup table for user_id to zero indexed number
    return users, items
