import datetime
import itertools
import json
import multiprocessing
import os
from collections import namedtuple
from random import shuffle

import gensim
import numpy as np
import progressbar
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

from rec_si.utils import elapsed_timer, easy_parallize

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

MovieDocument = namedtuple('MovieDocument', 'words tags split')

word_punct_tokenizer = WordPunctTokenizer()
stopwords = stopwords.words('english')


def clean_tokenizer(doc):
    return [word for word in word_punct_tokenizer.tokenize(doc) if word not in stopwords]


def iter_documents(subs_processed_path, doc_paths): # stream
    """
    Generator: iterate over all relevant documents, yielding one
    document at a time.
    """
    for doc_fname in doc_paths:
        with open(os.path.join(subs_processed_path, doc_fname)) as d:
            yield d.read(), doc_fname


class DocIterator(object):
    def __init__(self, subs_path, doc_paths, test_paths, tokenizer):
        self.subs_path = subs_path
        self.doc_paths = doc_paths
        self.test_paths = test_paths
        self.tokenizer = tokenizer

    def __iter__(self):
        sub_processed_path = os.path.join(self.subs_path, 'processed')
        for text, doc_fname in iter_documents(sub_processed_path, self.doc_paths):
            words = self.tokenizer(text)
            split = 'test' if doc_fname in self.test_paths else 'train'
            yield MovieDocument(words, [doc_fname], split) # filename as tag


def get_train_test_split(subs_path, test_size=0.2):
    if os.path.isfile(subs_path + 'all.txt') and os.path.isfile(subs_path + 'train.txt') and os.path.isfile('test.txt'):
        all_paths = np.loadtxt(subs_path + 'all.txt', dtype=np.str)
        train_paths = np.loadtxt(subs_path + 'train.txt', dtype=np.str)
        test_paths = np.loadtxt(subs_path + 'test.txt', dtype=np.str)

    else:
        from sklearn.cross_validation import train_test_split

        all_paths = np.array(os.listdir(subs_path + 'processed/')) # filename as label
        train_paths, test_paths = train_test_split(all_paths, test_size=test_size, random_state=42)

        np.savetxt(subs_path + 'all.txt', all_paths, fmt='%s')
        np.savetxt(subs_path + 'train.txt', train_paths, fmt='%s')
        np.savetxt(subs_path + 'test.txt', test_paths, fmt='%s')

    return all_paths, train_paths, test_paths


def read_docs(args):  # into RAM
    params, q = args
    all_paths, test_paths, subs_path, tokenizer = params

    docs = []

    for i, doc_fname in enumerate(all_paths):  # filename as label
        with open(os.path.join(subs_path, 'processed', doc_fname)) as d:
            words = tokenizer(d.read())
            split = 'test' if doc_fname in test_paths else 'train'
            docs.append(MovieDocument(words, [doc_fname], split))
            if q is not None:
                q.put(1)

    return docs


def get_docs(subs_path, tokenizer, stream=False):
    all_paths, train_paths, test_paths = get_train_test_split(subs_path, test_size=0.2)

    if stream:
        docs = DocIterator(subs_path, all_paths, test_paths, tokenizer)
    else:
        print "Loading docs into RAM..."

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        params = zip(chunks(all_paths, len(all_paths)/multiprocessing.cpu_count()),
                     itertools.repeat(test_paths),
                     itertools.repeat(subs_path),
                     itertools.repeat(tokenizer))

        result = easy_parallize(read_docs, params, p=multiprocessing.cpu_count(), nb_jobs=len(all_paths))
        docs = [doc for doc_arr in result for doc in doc_arr]

    return docs


def train_doc2vec(config, all_docs, train_docs=None, test_docs=None):
    if 'cores' in config:
        cores = config['cores']
    else:
        cores = multiprocessing.cpu_count()

    # create model
    model = Doc2Vec(size=config['size'], window=config['window'], min_count=config['min_count'],
                    workers=cores, alpha=config['alpha'], min_alpha=config['min_alpha'], dm=config['dm'],
                    dm_mean=config['dm_mean'], dm_concat=config['dm_concat'],
                    negative=config['negative'], hs=config['hs'])

    if 'train_words' in config and config['train_words']:
        model.train_words = True
    else:
        model.train_words = False

    print "Building vocabulary ..."
    with elapsed_timer() as elapsed:
        model.build_vocab(all_docs)
        print 'Building vocabulary took %.1f' % elapsed()

    print "Start training ..."
    bar = progressbar.ProgressBar(max_value=config['nb_epochs'])
    bar.start()
    for epoch in range(config['nb_epochs']):
        bar.update(epoch)

        # shuffle
        shuffle(all_docs)

        # train
        model.train(all_docs)

        # evaluate
        if 'eval' in config and config['eval'] and train_docs is not None and test_docs is not None:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                #err, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
                eval_duration = '%.1f' % eval_elapsed()

            #print("%f : epoch: %i duration: %ss eval duration: %ss" % (err, epoch, duration, eval_duration))

        if 'alpha_delta' in config:
            model.alpha -= config['alpha_delta'] # decrease the learning rate
            model.min_alpha = model.alpha # fix the learning rate, no decay
        elif 'alpha_decay' in config:
            model.alpha *= (1.0 - config['alpha_decay'])
            model.min_alpha = model.alpha

    bar.finish()

    print "Saving model ..."
    dt = datetime.datetime.now()
    model.save('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}'.format(dt, config['experiment_name']))
    with open('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'.format(dt, config['experiment_name']), 'w') as f:
        f.write(json.dumps(config))


if __name__ == "__main__":
    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    subs_path = 'data/subs/'

    print("START %s" % datetime.datetime.now())

    print "Get documents..."
    all_docs = get_docs(subs_path, clean_tokenizer, stream=False)

    config = {}
    config['size'] = 20
    config['window'] = 8
    config['min_count'] = 2
    config['alpha'] = 0.025
    config['min_alpha'] = 0.001
    config['dm'] = 0
    config['dm_mean'] = 0
    config['dm_concat'] = 0
    config['negative'] = 10
    config['hs'] = 0
    config['nb_epochs'] = 20
    config['train_words'] = False

    config['alpha_delta'] = (config['alpha'] - config['min_alpha']) / config['nb_epochs']
    #config['alpha_decay'] = 3e-2

    config['experiment_name'] = '20e_pv-dbow_size20_window8_neg10'
    config['eval'] = False
    config['cores'] = 4
    train_doc2vec(config, all_docs)

    print("END %s" % str(datetime.datetime.now()))
