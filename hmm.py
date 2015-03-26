"""
Created on Feb 13 2015
@author Wei Wang

Implement the base class for HMM
The implementation is based on:
    - Input-Output HMM's for Sequence processing
    - scikit-learn HMM implementation
"""

from __future__ import division
__author__ = 'Wei Wang'
__email__ = 'tskatom@vt.edu'

import numpy as np
from util import extmath
import string


class Abstract_hmm(object):
    def __init__(self, n_components, algorithm="viterbi", n_iter=70, thresh=1e-2, params=string.ascii_letters):
        """
        Initiate the model parameters
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.thresh = thresh
        self.algorithm = algorithm
        self.params = params

        # initiate the transition probability
        self.trans_mat = np.zeros((self.n_components, self.n_components))

        # initiate the start probability
        self.start_prob = np.zeros(self.n_components)

        # initiate the sufficient statistic object
        self.stats = {}

    def initiate_sufficient_statics(self):
        self.stats = {'nobs': 0,
                      'start': np.zeros(self.n_components),
                      'trans': np.zeros((self.n_components, self.n_components))}

    def compute_forward_lattice(self, log_obs_likelihood):
        """
        Compute Alpha the forward lattice.
        """
        n_obs, n_components = log_obs_likelihood.shape
        fwdlattice = np.zeros((n_obs, n_components))
        log_trans_mat = np.log(self.trans_mat)
        fwdlattice[0] = log_obs_likelihood[0] + np.log(self.start_prob)

        for t in range(1, n_obs):
            for i in range(n_components):
                work_buffer = []
                for j in range(n_components):
                    work_buffer.append(fwdlattice[t-1, j] + log_trans_mat[j, i])
                fwdlattice[t, i] = log_obs_likelihood[t, i] + extmath.logsumexp(work_buffer)
        lpr = extmath.logsumexp(fwdlattice[-1])
        return lpr, fwdlattice

    def compute_backward_lattice(self, log_obs_likelihood):
        """
        Compute Beta the backward lattice
        """
        n_obs, n_components = log_obs_likelihood.shape
        bwdfattice = np.zeros((n_obs, n_components))
        bwdfattice[n_obs-1] = np.zeros(n_components)
        log_trans_mat = np.log(self.trans_mat)
        for t in range(n_obs-2, -1, -1):
            for i in range(n_components):
                work_buffer = []
                for j in range(n_components):
                    work_buffer.append(log_trans_mat[i,j] + log_obs_likelihood[t+1,j] + bwdfattice[t+1, j])
                bwdfattice[t, i] = extmath.logsumexp(work_buffer)

        return bwdfattice

    def do_maxstep(self, params):
        """
        do the maximization step of the hmm
        """
        # in the base model we will implemented the maximization for start probability and transition probability
        if "s" in params:
            self.start_prob = extmath.normalize(np.maximum(self.stats['start'], 1e-20))

        if "t" in params:
            self.trans_mat = extmath.normalize(np.maximum(self.stats['trans'], 1e-20), axis=1)


class Hmm(Abstract_hmm):
    def init(self, obs, params):
        self.trans_mat.fill(1.0 / self.n_components)
        self.start_prob.fill(1.0 / self.n_components)

    def fit(self, obs):
        """
        Fit the model
        """
        log_prob = [] # this is the log probability of the whole observations
        self.init(obs, self.params)
        for n in range(self.n_iter):
            curr_log_prob = .0
            self.initiate_sufficient_statics()
            for p in range(len(obs)):
                obs_seq = obs[p]
                log_obs_likelihood = self.compute_log_obs_likelihood(obs_seq)
                lpr, fwdlattice = self.compute_forward_lattice(log_obs_likelihood)
                bwdlattice = self.compute_backward_lattice(log_obs_likelihood)
                curr_log_prob += lpr
                gamma = fwdlattice + bwdlattice
                posterior = np.exp(gamma - lpr)
                self.accum_sufficient_statics(obs_seq, log_obs_likelihood, posterior, fwdlattice, bwdlattice, self.params)
            log_prob.append(curr_log_prob)

            # check for convergence
            if n > 0 and log_prob[-1] - log_prob[-2] < self.thresh:
                break

            # do the maximization step
            self.do_maxstep(self.params)
        return self

    def compute_log_obs_likelihood(self, obs_seq):
        """
        compute the log likelihood of the observation in each time step.
        The method should be implemented in the sub class
        """
        raise Exception("compute_log_obs_likelihood must be implemented!")
        log_obs_likelihood = np.zeros(len(obs_seq), self.n_components)
        return log_obs_likelihood

    def accum_sufficient_statics(self, obs_seq, log_obs_likelihood, posterior, fwdlattice, bwdlattice, params):
        """
        compute the statistic sufficients for maximization step
        """
        self.stats['nobs'] += 1
        lpr = extmath.logsumexp(fwdlattice[-1])
        log_trans_mat = np.log(self.trans_mat)

        n_obs, n_components = log_obs_likelihood.shape
        if 's' in params:
            self.stats['start'] += posterior[0]

        if 't' in params:
            # we firstly compute the joint posteriors p(t=i,t+1=j|X)
            if n_obs <= 1:
                return
            joint_post = np.asarray(np.tile(.0, (n_obs-1, n_components, n_components)))
            for t in range(n_obs-1):
                for i in range(n_components):
                    for j in range(n_components):
                        joint_post[t, i, j] = np.exp(fwdlattice[t, i] +
                                                     log_obs_likelihood[t+1, j] +
                                                     log_trans_mat[i, j] +
                                                     bwdlattice[t+1, j] - lpr)
                self.stats['trans'] += joint_post[t,:,:]

    def decode(self, obs_seq, algorithm="viterbi"):
        """
        find the most probable sequence
        """
        if algorithm == "viterbi":
            sequence = self.viterbi(obs_seq)
        return sequence

    def viterbi(self, obs_seq):
        log_trans_mat = np.log(self.trans_mat)
        log_obs_likelihood = self.compute_log_obs_likelihood(obs_seq)
        n_obs, n_components = log_obs_likelihood.shape
        v_log_forward = np.zeros((n_obs, n_components))
        a_induction = np.zeros((n_obs, n_components), dtype=np.int) # the most probable previous status
        sequences = np.zeros(n_obs, dtype=np.int)
        v_log_forward[0] = np.log(self.start_prob) + log_obs_likelihood[0]

        for t in range(1, n_obs):
            for i in range(n_components):
                work_buffer = v_log_forward[t-1] + log_trans_mat[:,i]
                v_log_forward[t, i] = np.max(work_buffer) + log_obs_likelihood[t, i]
                a_induction[t, i] = np.argmax(work_buffer)

        # induce the most likely sequences
        sequences[n_obs-1] = np.argmax(v_log_forward[n_obs-1])
        for t in range(n_obs-2, -1, -1):
            sequences[t] = a_induction[t+1, sequences[t+1]]

        return sequences

class OHmm(Abstract_hmm):
    """
    The extention to basic HMM which support add covariates into the emission part
    """
    def init(self, ins, obs, params):
        self.trans_mat.fill(1.0 / self.n_components)
        self.start_prob.fill(1.0 / self.n_components)

    def fit(self, ins, obs):
        """
        Fit the model
        """
        log_prob = [] # this is the log probability of the whole observations
        self.init(ins, obs, self.params)
        for n in range(self.n_iter):
            curr_log_prob = .0
            self.initiate_sufficient_statics()
            for p in range(len(obs)):
                ins_seq = ins[p]
                obs_seq = obs[p]
                log_obs_likelihood = self.compute_log_obs_likelihood(ins_seq, obs_seq)
                lpr, fwdlattice = self.compute_forward_lattice(log_obs_likelihood)
                bwdlattice = self.compute_backward_lattice(log_obs_likelihood)
                curr_log_prob += lpr
                gamma = fwdlattice + bwdlattice
                posterior = np.exp(gamma - lpr)
                self.accum_sufficient_statics(ins_seq, obs_seq, log_obs_likelihood, posterior, fwdlattice, bwdlattice, self.params)
            log_prob.append(curr_log_prob)

            print '[------------Likelihood------------', n,'--]', curr_log_prob
            # check for convergence
            if n > 0 and abs(log_prob[-1] - log_prob[-2]) < self.thresh:
                print "Converged"
                break

            # do the maximization step
            self.do_maxstep(self.params)
        return self

    def compute_log_obs_likelihood(self, ins_seq, obs_seq):
        """
        compute the log likelihood of the observation in each time step.
        The method should be implemented in the sub class
        """
        log_obs_likelihood = np.zeros(len(obs_seq), self.n_components)
        raise Exception("compute_log_obs_likelihood must be implemented!")
        return log_obs_likelihood

    def accum_sufficient_statics(self, ins_seq, obs_seq, log_obs_likelihood, posterior, fwdlattice, bwdlattice, params):
        """
        compute the statistic sufficients for maximization step
        """
        self.stats['nobs'] += 1
        lpr = extmath.logsumexp(fwdlattice[-1])
        log_trans_mat = np.log(self.trans_mat)

        n_obs, n_components = log_obs_likelihood.shape
        if 's' in params:
            self.stats['start'] += posterior[0]

        if 't' in params:
            # we firstly compute the joint posteriors p(t=i,t+1=j|X)
            if n_obs <= 1:
                return
            joint_post = np.asarray(np.tile(.0, (n_obs-1, n_components, n_components)))
            for t in range(n_obs-1):
                for i in range(n_components):
                    for j in range(n_components):
                        joint_post[t, i, j] = np.exp(fwdlattice[t, i] +
                                                     log_obs_likelihood[t+1, j] +
                                                     log_trans_mat[i, j] +
                                                     bwdlattice[t+1, j] - lpr)
                self.stats['trans'] += joint_post[t,:,:]

    def decode(self, ins_seq, obs_seq, algorithm="viterbi"):
        """
        find the most probable sequence
        """
        if algorithm == "viterbi":
            sequence = self.viterbi(ins_seq, obs_seq)
        return sequence

    def viterbi(self, ins_seq, obs_seq):
        log_trans_mat = np.log(self.trans_mat)
        log_obs_likelihood = self.compute_log_obs_likelihood(ins_seq, obs_seq)
        n_obs, n_components = log_obs_likelihood.shape
        v_log_forward = np.zeros((n_obs, n_components))
        a_induction = np.zeros((n_obs, n_components), dtype=np.int) # the most probable previous status
        sequences = np.zeros(n_obs, dtype=np.int)
        v_log_forward[0] = np.log(self.start_prob) + log_obs_likelihood[0]

        for t in range(1, n_obs):
            for i in range(n_components):
                work_buffer = v_log_forward[t-1] + log_trans_mat[:,i]
                v_log_forward[t, i] = np.max(work_buffer) + log_obs_likelihood[t, i]
                a_induction[t, i] = np.argmax(work_buffer)

        # induce the most likely sequences
        sequences[n_obs-1] = np.argmax(v_log_forward[n_obs-1])
        for t in range(n_obs-2, -1, -1):
            sequences[t] = a_induction[t+1, sequences[t+1]]

        print "log_obs_likelihood", np.exp(log_obs_likelihood)
        return sequences