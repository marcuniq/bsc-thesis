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

from utils import elapsed_timer


assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

MovieDocument = namedtuple('MovieDocument', 'words tags split')


def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    #print(predictor.summary())
    return predictor


def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.tags[0], test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.tags[0] for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


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

        all_paths = np.array(os.listdir(subs_path + 'processed\\')) # filename as label
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
        point = total / 100
        increment = total / 20

        for i, doc_fname in enumerate(all_paths): # filename as label
            if(i % (5 * point) == 0):
                sys.stdout.write("\r[" + "=" * (i / increment) +  " " * ((total - i)/ increment) + "]" + str(i / point) + "%")
                sys.stdout.flush()
            with open(os.path.join(subs_path, 'processed', doc_fname)) as d:
                words = tokenizer(d.read())
                split = 'test' if doc_fname in test_paths else 'train'
                docs.append(MovieDocument(words,[doc_fname], split))
        print ""
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
    if 'cores' in config.keys():
        cores = config['cores']
    else:
        cores = multiprocessing.cpu_count()

    # create model
    model = Doc2Vec(size=config['size'], window=config['window'], min_count=config['min_count'],
                    workers=cores, alpha=config['alpha'], min_alpha=config['min_alpha'], dm=config['dm'],
                    negative=config['negative'], hs=config['hs']) # use fixed learning rate

    if 'train_words' in config.keys() and config['train_words']:
        model.train_words = True
    else:
        model.train_words = False

    print "Building vocabulary ..."
    with elapsed_timer() as elapsed:
        model.build_vocab(all_docs)
        print 'Building vocabulary took %.1f' % elapsed()

    print "Start training ..."
    for epoch in range(config['nb_epochs']):
        print "epoch {}".format(epoch)

        # train
        with elapsed_timer() as elapsed:
            model.train(all_docs)
            duration = '%.1f' % elapsed()

        # evaluate
        if 'eval' in config.keys() and config['eval'] and train_docs and test_docs:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
                eval_duration = '%.1f' % eval_elapsed()

            print("%f : epoch: %i duration: %ss eval duration: %ss" % (err, epoch, duration, eval_duration))

        if 'alpha_delta' in config.keys():
            model.alpha -= config['alpha_delta'] # decrease the learning rate
            model.min_alpha = model.alpha # fix the learning rate, no decay

        print "Saving model ..."
        dt = datetime.datetime.now()
        model.save('doc2vec-models\\{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}'.format(dt, config['experiment_name'], epoch))
        with open('doc2vec-models\\{:%Y-%m-%d_%H.%M.%S}_{}_epoch{}_config.json'
                          .format(dt, config['experiment_name'], epoch), 'w') as f:
            f.write(json.dumps(config))


if __name__ == "__main__":
    subs_path = 'data\\subs\\'

    print("START %s" % datetime.datetime.now())

    print "Get documents..."
    with elapsed_timer() as elapsed:
        all_docs, train_docs, test_docs = get_docs(subs_path, stream=False)
        print 'Get documents took %.1f' % elapsed()

    config = {'size': 100, 'window': 5, 'min_count': 2, 'alpha': 0.05, 'min_alpha': 0.05,
              'dm': 0, 'negative': 2, 'hs': 0,
              'nb_epochs':2, 'alpha_delta': 0.002,
              'experiment_name': 'test', 'eval': True}
    train_doc2vec(config, all_docs, train_docs, test_docs)

    print("END %s" % str(datetime.datetime.now()))
