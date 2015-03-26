__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'

import sys
from hmm import OHmm
import string
import numpy as np
from poisson_regression import PoissonRegression
from util import extmath
from matplotlib import pyplot as pl

class PoissonHmm(OHmm):
    def __init__(self, n_components, algorithm="viterbi", n_iter=500,
                 thresh=1e-2, params=string.ascii_letters, p_rate=0.3):
        super(PoissonHmm, self).__init__(n_components, algorithm,
                                         n_iter, thresh, params)
        self.p_rate = p_rate

    def init(self, ins, obs, params="stp"):
        super(PoissonHmm, self).init(ins, obs, params)
        # normalize the external features
        if not hasattr(self, "n_features"):
            self.n_features = ins[0].shape[1]

        if "p" in params:
            # initiated parameters
            features = np.vstack(ins)
            targets = np.vstack(obs)
            poss = PoissonRegression(features, targets)
            poss.fit()
            self.obs_beta_mat = np.tile(np.squeeze(np.asarray(poss.beta)),
                                        (self.n_components, 1))
            self.obs_beta_mat = np.random.rand(self.n_components, self.n_features)
        self.ins = ins
        self.obs = obs

    def compute_log_obs_likelihood(self, ins_seq, obs_seq):
        nu = np.dot(ins_seq, self.obs_beta_mat.T)
        mu = np.exp(nu)
        obs_seq = obs_seq[np.newaxis].T
        log_y_fac = np.asarray(map(extmath.logfac, obs_seq))
        return nu * obs_seq - mu - log_y_fac[np.newaxis].T

    def initiate_sufficient_statics(self):
        super(PoissonHmm, self).initiate_sufficient_statics()
        self.stats['posteriors'] = []

    def accum_sufficient_statics(self, ins_seq, obs_seq, log_obs_likelihood, posterior,
                                 fwdlattice, bwdlattice, params):
        super(PoissonHmm, self).accum_sufficient_statics(ins_seq, obs_seq, log_obs_likelihood,
                                                         posterior, fwdlattice, bwdlattice, params)

        self.stats['posteriors'].append(posterior)

    def do_maxstep(self, params):
        super(PoissonHmm, self).do_maxstep(params)

        # we have to use numerical solution to get the parameters for Poission Regression
        for j in range(self.n_components):
            self.optimize_obs_beta(self.ins, self.obs, j, 40)

    def optimize_obs_beta(self, ins, obs, j, n_iter, threshold=1e-6):
        obs_beta = self.obs_beta_mat[j].reshape(self.n_features, 1) # n_features * 1
        difference = []
        for n in range(n_iter):
            jac_vec = np.zeros(self.n_features)
            hess = np.zeros((self.n_features, self.n_features))
            for p in range(len(ins)):
                ins_seq = ins[p]
                obs_seq = obs[p]
                obs_seq = obs_seq.reshape(len(obs_seq), 1)
                posts = self.stats['posteriors'][p][:, j].reshape(len(ins_seq), 1) # T * 1
                nu = ins_seq.dot(obs_beta)
                mu = np.exp(nu)
                z = obs_seq - mu
                jac_vec += np.sum(posts * z * ins_seq, axis=0)
                diag_mu = np.diag(np.squeeze(mu))
                hess += -1 * ins_seq.T.dot(diag_mu).dot(ins_seq)

            jac_vec = jac_vec - 2 * self.p_rate * np.squeeze(obs_beta)
            hess = hess - 2 * self.p_rate * np.identity(self.n_features)
            beta_old = obs_beta
            try:
                obs_beta = obs_beta - np.linalg.pinv(hess).dot(jac_vec[np.newaxis].T)
            except Exception as e:
                print 'Error-----------'
                sys.exit()
            """
            # check the step size
            pre_obj = self.obj_obs_subnet(beta_old, j)
            after_obj = self.obj_obs_subnet(obs_beta, j)
            delta = 0.9 * np.linalg.pinv(hess).dot(jac_vec[np.newaxis].T)
            while after_obj < pre_obj:
                obs_beta = beta_old - delta
                delta *= 0.9
                pre_obj = self.obj_obs_subnet(beta_old, j)
                after_obj = self.obj_obs_subnet(obs_beta, j)
            """
            difference.append(np.max(beta_old - obs_beta))
            if difference[-1] <= threshold:
                break
        self.obs_beta_mat[j, :] = np.squeeze(obs_beta)

    def obj_obs_subnet(self, beta, j):
        # maximize the subnetwork one by one
        # theta is a D * 1 vectorz
        obj = 0.0
        for p in range(len(self.obs)):
            obs_seq = self.obs[p][np.newaxis].T
            ins_seq = self.ins[p]
            posts = self.stats['posteriors'][p][:, j].reshape(len(ins_seq), 1)
            nu = np.dot(ins_seq, beta)
            mu = np.exp(nu)
            log_y_fac = np.asarray(map(extmath.logfac, obs_seq))[np.newaxis].T
            z = obs_seq * nu - log_y_fac - mu
            obj += np.sum(posts * z)
        return float(obj)

    def one_step_predict(self, ins_seq, obs_seq, u, method='max'):
        """
        If the method = 'max', we only use the most probable status to make prediction
        if the method = 'weighted', we use the expectation as prediction
        params:
        ins_seq: n * d array, d is the feature dimension
        obs_seq: n * 1 array
        u: d * 1 array
        """
        # compute the most likely future status
        log_trans_mat = np.log(self.trans_mat)
        log_obs_likelihood = self.compute_log_obs_likelihood(ins_seq, obs_seq)
        n_obs, n_components = log_obs_likelihood.shape
        v_log_forward = np.zeros((n_obs, n_components))
        a_induction = np.zeros((n_obs, n_components), dtype=np.int) # the most probable previous status
        v_log_forward[0] = np.log(self.start_prob) + log_obs_likelihood[0]

        for t in range(1, n_obs):
            for i in range(n_components):
                work_buffer = v_log_forward[t-1] + log_trans_mat[:,i]
                v_log_forward[t, i] = np.max(work_buffer) + log_obs_likelihood[t, i]
                a_induction[t, i] = np.argmax(work_buffer)

        next_status_prob = np.zeros(self.n_components, dtype=np.int)
        for i in range(n_components):
            work_buffer = v_log_forward[-1] + log_trans_mat[:, i]
            next_status_prob[i] = np.max(work_buffer)

        if method == 'max':
            pred_status = np.argmax(next_status_prob)
            prediction = int(np.exp(np.dot(self.obs_beta_mat[pred_status], u)))

        elif method == 'weighted':
            weights = np.exp(next_status_prob - np.sum(next_status_prob))
            prediction = int(np.dot(weights, np.exp(np.dot(self.obs_beta_mat, u[np.newaxis].T))))
        return prediction

if __name__ == "__main__":
    # generate fake examples
    np.random.seed(0)
    status = [0, 1, 2, 3] * 50
    real_beta = np.array([[3.5, 0.1], [0.1, 3.5], [2, 2], [0.1, 0.1]])
    obs = []
    ins = []
    lams = []
    for s in status:
        x = np.hstack([[1.0], np.random.rand(1)])
        ins.append(x)
        lam = np.exp(np.dot(x, real_beta[s]))
        y = np.random.poisson(lam)
        obs.append(y)
        lams.append(lam)

    print "INS", ins
    print "OBS", obs
    print "lams", lams
    print "Real Beta", real_beta
    print "n_components: ", 4

    ins = np.array(ins)
    obs = np.array(obs)

    phmm = PoissonHmm(n_components=4)
    print "Previous", phmm.obs_beta_mat
    phmm.fit([ins], [obs])
    print "Updated", phmm.obs_beta_mat
    print "Transition", phmm.trans_mat
    print "Start", phmm.start_prob

    sequence = phmm.decode(ins, obs)
    fig = pl.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    xs = np.arange(len(obs))
    for i in range(4):
        ix = (sequence == i)
        ax.plot(xs[ix], obs[ix], 'o', label='status %d' % i)
    ax.plot(xs, obs, '-', label='Line', color='black')
    ax.legend()
    pl.show()
