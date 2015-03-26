__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'

import pandas as pds
import numpy as np
import os
from matplotlib import pyplot as pl
from pohmm import PoissonHmm
from multiprocessing import Process, Pool
import cProfile

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

    n_components = 3
    pmm = PoissonHmm(n_components, n_iter=1000)

    # normalize trainingX
    mean_x = np.mean(trainX, axis=0)
    std_x = np.std(trainX, axis=0)
    norm_trainX = (trainX - mean_x) / std_x
    pmm.fit([norm_trainX], [trainY])
    print pmm.obs_beta_mat
    print pmm.start_prob
    print pmm.trans_mat
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
    pl.show()

def one_step_pred(n_components, observations):
    # observations is a array like object. n * d
    if len(observations.shape) == 1:
        observations = observations[np.newaxis].T
    pmm = PoissonHmm(n_components, n_iter=1000)
    pmm.fit([observations])
    preds = pmm.predict(observations)
    ix = np.argmax(pmm.transmat_[preds[-1]])
    pred_v = pmm.means_[ix]
    return pred_v

def test(icews_file, event_type, test_num=20):
    events_count = pds.DataFrame.from_csv(icews_file, sep='\t', index_col=1)
    del events_count['20']
    del events_count['country']
    events_count = events_count.sort_index()['2012-01-01':]
    events_count = events_count.resample('W', how='sum').fillna(0)
    target = event_type
    # construct the training and test set
    Ys = events_count[target].values
    testYs = Ys[-test_num:]
    preds = []
    for i in range(-test_num, 0, 1):
        trainY = Ys[:i]
        try:
            pred = one_step_pred(n_components=5, observations=trainY)
            preds.append(int(pred))
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

def demo2(icews_file, event_type):
    print icews_file, event_type

def main():
    import glob
    steps = ["demo3"]
    icews_folder = "/home/weiwang/To_Jieping/icews_allevent_count/"
    icews_files = glob.glob(icews_folder + "*")
    for step in steps:
        if step == "demo":
            for f in icews_files:
                event_type = "14"
                demo(f, event_type)
            pl.show()
        elif step == "test":
            performance = {}
            details = {}
            for f in icews_files:
                event_type = "14"
                testYs, preds, scores = test(f, event_type)
                basename = os.path.basename(f).split("_icews")[0]
                print testYs, preds, scores
                print basename, np.mean(scores)
                performance[basename] = [np.mean(scores)]
                details[basename] = {"truth": list(testYs), "preds": list(preds), "scores": list(scores)}
            perf_pds = pds.Series(performance)
            detail_pds = pds.DataFrame(details)
            perf_pds.to_csv('./data/perform.csv')
            detail_pds.to_csv('./data/detail.csv')
        elif step == "demo2":
            p = Pool(14)
            zip(icews_files, ["14"]*len(icews_files))
            rs = [p.apply_async(demo, args=(f, e)) for f, e in zip(icews_files, ["14"]*len(icews_files))]
            for r in rs:
                r.get()
            pl.show()
        elif step == "demo3":
            f = icews_files[0]
            cProfile.run("demo('/home/weiwang/To_Jieping/icews_allevent_count/syria_icews_parent_event_counts_2012_228.csv', '14')")
main()