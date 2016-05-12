#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def binarize_ratings(df, pos=1, neg=0, threshold=1):
    copy = df.copy()
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


def easy_parallize(f, params, p=8):
    from multiprocessing import Pool, Manager
    pool = Pool(processes=p)
    m = Manager()
    q = m.Queue()

    nb_jobs = len(params)
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
