import datetime
import json
import os
import pandas as pd

from metrics import run_eval


def train_eval_save(config, train_func):
    model, config, loss_history = train_func(config)
    dt = datetime.datetime.now()

    if config['verbose'] > 0:
        print "Saving model..."

    model_save_dir = os.path.abspath(config['model_save_dir'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model.save(os.path.join(model_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}.h5'.format(dt, config['experiment_name'])))
    with open(os.path.join(model_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}_config.json'.format(dt, config['experiment_name'])), 'w') as f:
        f.write(json.dumps(config))

    if loss_history is not None:
        with open(os.path.join(model_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}_history.json'.format(dt, config['experiment_name'])), 'w') as f:
            f.write(json.dumps(loss_history))

    if config['run_eval']:
        ratings = pd.read_csv(config['ratings_path'])
        train = pd.read_csv(config['train_path'])
        if config['val']:
            test = pd.read_csv(config['val_path'])
        else:
            test = pd.read_csv(config['test_path'])

        user_metrics, top_n_predictions, movie_metrics = run_eval(model, train, test, ratings, config)

        if config['verbose'] > 0:
            print "Saving user metrics and predictions ..."

        metrics_save_dir = os.path.abspath(config['metrics_save_dir'])
        if not os.path.exists(metrics_save_dir):
            os.makedirs(metrics_save_dir)

        user_metrics.to_csv(os.path.join(metrics_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}_user-metrics.csv'.format(dt, config['experiment_name'])), index=False)
        top_n_predictions.to_csv(os.path.join(metrics_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}_top-{}-predictions.csv'.format(dt, config['experiment_name'], config['top_n_predictions'])), index=False)

        if movie_metrics is not None:
            movie_metrics.to_csv(os.path.join(metrics_save_dir, '{:%Y-%m-%d_%H.%M.%S}_{}_movie-metrics.csv'.format(dt, config['experiment_name'])), index=False)
