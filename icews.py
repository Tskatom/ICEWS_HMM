__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'

import pandas as pds
import numpy as np
import os
from matplotlib import pyplot as pl
from pohmm import PoissonHmm
import sys

def demo(icews_file, event_type):
    events_count = pds.DataFrame.from_csv(icews_file, sep='\t', index_col=1)
    del events_count['20']
    del events_count['country']
    events_count = events_count.sort_index()['2012-01-01':]
    events_count = events_count.resample('W', how='sum').fillna(0)
    columns = events_count.columns
    target = event_type
    excepts = ["14"]
    features = [col for col in columns if col not in excepts]

    # construct the training and test set
    Ys = events_count[target]
    Xs = events_count[features]

    trainX = Xs['2012-01-08':'2015-02-15'].values
    trainY = Ys['2012-01-15':'2015-02-22'].values
    datesY = Ys['2012-01-15':'2015-02-22'].index

    n_components = 5
    pmm = PoissonHmm(n_components, n_iter=1000)

    # normalize trainingX
    mean_x = np.mean(trainX, axis=0)
    std_x = np.std(trainX, axis=0)
    norm_trainX = (trainX - mean_x) / std_x
    pmm.fit([norm_trainX], [trainY])
    sequence = pmm.decode(norm_trainX, trainY)

    # plot the process
    fig = pl.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    for i in range(n_components):
        ix = (sequence == i)
        ax.plot_date(datesY[ix], trainY[ix], 'o', label="%dth hidden state" % i)
    ax.plot(Ys.index, Ys, '-', label='%s event' % event_type)
    ax.legend()
    basename = os.path.basename(icews_file).split('_icews')[0]
    ax.set_title('%s' % basename)
    pl.draw()

def pohmm_experiment(icews_file, event_type, test_num=20):
    events_count = pds.DataFrame.from_csv(icews_file, sep='\t', index_col=1)
    del events_count['20']
    del events_count['country']
    start = '2012-01-02'
    end = '2015-03-22'
    date_range = pds.date_range(start, end)
    events_count = events_count.reindex(date_range).fillna(0)
    events_count = events_count.sort_index()['2012-01-02':]
    events_count = events_count.resample('W', how='sum').fillna(0)
    target = event_type

    #features = [c for c in events_count.columns if c != target]
    features = ["14", "17", "18"]
    # construct the training and test set
    Xs = events_count[features].values[:-1,:]
    mean_x = np.mean(Xs, axis=0)
    std_x = np.std(Xs, axis=0)
    norm_xs = (Xs - mean_x) / std_x
    # add dumpy 1 column in x
    ones = np.ones((len(norm_xs), 1))
    norm_xs = np.hstack([ones, norm_xs])

    Ys = events_count[target].values[1:]
    testYs = Ys[-test_num:]
    preds = []
    basename = os.path.basename(icews_file).split('_icews')[0]
    for i in range(-test_num, 0, 1):
        trainX = norm_xs[:i,:]
        trainY = Ys[:i]
        try:
            phmm = PoissonHmm(n_components=4)
            phmm.fit([trainX], [trainY])
            pred = phmm.one_step_predict(trainX, trainY, norm_xs[i])
            preds.append(int(pred))
            print "Round: %d ____  %s Pred: %d, Truth: %d, score: %0.2f " % (-1*i, basename, pred, Ys[i], score(pred, Ys[i]))
        except:
            print 'Exception', icews_file
            return [],[],[]
    return testYs, preds, map(score, testYs, preds)

def score(pred, truth):
    occu = (pred > 0) == (truth > 0)
    pred = float(pred)
    truth = float(truth)
    accu = 1 - abs(pred - truth) / max([pred, truth, 4.0])
    return 0.5 * occu + 3.5 * accu

def main():
    import glob
    steps = ["pohmm_experiment"]
    icews_folder = "/home/weiw/workspace/data/icews/232/"
    icews_files = glob.glob(icews_folder + "*icews_parent_event_counts*.csv")
    for step in steps:
        if step == "demo":
            for f in icews_files:
                event_type = "14"
                demo(f, event_type)
            pl.show()
        elif step == "pohmm_experiment":
            performance = {}
            details = {}
            for f in icews_files:
                event_type = "14"
                testYs, preds, scores = pohmm_experiment(f, event_type)
                basename = os.path.basename(f).split("_icews")[0]
                print testYs, preds, scores
                print basename, np.mean(scores)
                performance[basename] = [np.mean(scores)]
                details[basename] = {"truth": list(testYs), "preds": list(preds), "scores": list(scores)}
            perf_pds = pds.Series(performance)
            detail_pds = pds.DataFrame(details)
            perf_pds.to_csv('./data/perform.csv')
            detail_pds.to_csv('./data/detail.csv')
main()