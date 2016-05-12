import os
import datetime
import json
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords
import multiprocessing
import numpy as np
import statsmodels.api as sm
from random import sample
from collections import namedtuple
import progressbar
from utils import elapsed_timer
from random import shuffle

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

MovieDocument = namedtuple('MovieDocument', 'words tags split')

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


def get_docs(subs_path, tokenizer=None, stream=False):
    from nltk.corpus import stopwords

    if not tokenizer:
        word_punct_tokenizer = WordPunctTokenizer()
        stopwords = stopwords.words('english')

        def clean_tokenizer(doc):
            return [word for word in word_punct_tokenizer.tokenize(doc) if word not in stopwords]

        tokenizer = clean_tokenizer

    MovieDocument = namedtuple('MovieDocument', 'words tags split')

    all_paths, train_paths, test_paths = get_train_test_split(subs_path, test_size=0.2)

    def create_doc_iterator(subs_path, labels, tokenizer, test_labels): # stream
        for doc_fname in labels: # filename as label
            with open(os.path.join(subs_path, 'processed', doc_fname)) as d:
                words = tokenizer(d.read())
                split = 'test' if doc_fname in test_labels else 'train'
                yield MovieDocument(words,[doc_fname], split)

    def create_doc_array(): # in RAM
        docs = []
        total = len(all_paths)
        bar = progressbar.ProgressBar(max_value=total)
        bar.start()

        for i, doc_fname in enumerate(all_paths): # filename as label
            bar.update(i)
            with open(os.path.join(subs_path, 'processed', doc_fname)) as d:
                words = tokenizer(d.read())
                split = 'test' if doc_fname in test_paths else 'train'
                docs.append(MovieDocument(words,[doc_fname], split))
        bar.finish()
        return docs

    if stream:
        docs = DocIterator(subs_path, all_paths, test_paths, tokenizer)
    else:
        print "Loading docs into RAM..."
        docs = create_doc_array()
    train_docs = (doc for doc in docs if doc.split == 'train')
    test_docs = (doc for doc in docs if doc.split == 'test')
    all_docs = docs

    return all_docs, train_docs, test_docs


def train_doc2vec(config, all_docs, train_docs=None, test_docs=None):
    if 'cores' in config:
        cores = config['cores']
    else:
        cores = multiprocessing.cpu_count()

    # create model
    model = Doc2Vec(size=config['size'], window=config['window'], min_count=config['min_count'],
                    workers=cores, alpha=config['alpha'], min_alpha=config['min_alpha'], dm=config['dm'],
                    dm_mean=config['dm_mean'], dm_concat=config['dm_concat'],
                    negative=config['negative'], hs=config['hs']) # use fixed learning rate

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

        # lr decay
        if 'alpha_decay' in config:
            model.alpha *= (1.0 - config['alpha_decay'])
            model.min_alpha = model.alpha

        # save
        if 'save_on_epoch_end' in config and config['save_on_epoch_end']:
            print "Saving model ..."
            dt = datetime.datetime.now()
            model.save('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}'.format(dt, config['experiment_name'], epoch))
            with open('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}_config.json'
                              .format(dt, config['experiment_name'], epoch), 'w') as f:
                f.write(json.dumps(config))

    bar.finish()
    if 'save_on_epoch_end' not in config or not config['save_on_epoch_end']:
        print "Saving model ..."
        dt = datetime.datetime.now()
        model.save('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}'.format(dt, config['experiment_name']))
        with open('doc2vec-models/{:%Y-%m-%d_%H.%M.%S}_{}_config.json'.format(dt, config['experiment_name']), 'w') as f:
            f.write(json.dumps(config))


if __name__ == "__main__":
    subs_path = 'data/subs/'

    print("START %s" % datetime.datetime.now())

    print "Get documents..."
    all_docs, train_docs, test_docs = get_docs(subs_path, stream=False)

    config = {'size': 50, 'window': 8, 'min_count': 2, 'alpha': 0.025, 'min_alpha': 0.025,
              'dm': 0, 'negative': 5, 'dm_mean': 0, 'dm_concat': 0, 'hs': 0,
              'nb_epochs': 100, 'alpha_decay': 3e-2, #'alpha_delta': 0.002,
              'experiment_name': '100e_pv-dbow_size50_lr0.025_decay_3e-2_window8_neg5', 'eval': True, 'save_on_epoch_end': False,
              'cores': 4}
    train_doc2vec(config, all_docs, train_docs, test_docs)

    print("END %s" % str(datetime.datetime.now()))
