__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'

import numpy as np
from hmm import Hmm
import string
from util import extmath
import testHmm

NEGINF = -np.inf
np.random.seed(1)

class MultinomialHMM(Hmm):
    def __init__(self, n_components=1, algorithm="viterbi",n_iter=10, thresh=1e-2, params=string.ascii_letters,):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of states.
        """
        Hmm.__init__(self, n_components, algorithm=algorithm, n_iter=n_iter, thresh=thresh, params=params)

    def _get_emissionprob(self):
        """Emission probability distribution for each state."""
        return np.exp(self._log_emissionprob)

    def _set_emissionprob(self, emissionprob):
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and \
                emissionprob.shape != (self.n_components, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_components, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            extmath.normalize(emissionprob)
        self._log_emissionprob = np.log(emissionprob)
        underflow_idx = np.isnan(self._log_emissionprob)
        self._log_emissionprob[underflow_idx] = NEGINF
        self.n_symbols = self._log_emissionprob.shape[1]

    emissionprob_ = property(_get_emissionprob, _set_emissionprob)

    def compute_log_obs_likelihood(self, obs):
        return self._log_emissionprob[:, obs].T

    def init(self, obs, params='ste'):
        super(MultinomialHMM, self).init(obs, params)
        if 'e' in params:
            if not hasattr(self, 'n_symbols'):
                symbols = set()
                for o in obs:
                    symbols = symbols.union(set(o))
                self.n_symbols = len(symbols)
            emissionprob = extmath.normalize(np.random.rand(self.n_components, self.n_symbols), 1)
            #emissionprob = np.array([[0.3, 0.7],[0.4, 0.6]])
            self.emissionprob_ = emissionprob

    def initiate_sufficient_statics(self):
        super(MultinomialHMM, self).initiate_sufficient_statics()
        self.stats['obs'] = np.zeros((self.n_components, self.n_symbols))

    def accum_sufficient_statics(self, obs_seq, framelogprob, posterior, fwdlattice, bwdlattice, params):
        super(MultinomialHMM, self).accum_sufficient_statics(obs_seq, framelogprob, posterior, fwdlattice, bwdlattice, params)
        if 'e' in params:
            for t, symbol in enumerate(obs_seq):
                self.stats['obs'][:, symbol] += posterior[t]

    def do_maxstep(self, params):
        super(MultinomialHMM, self).do_maxstep(params)
        if 'e' in params:
            self.emissionprob_ = (self.stats['obs']
                                  / self.stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, obs):
        """Check if ``obs`` is a sample from a Multinomial distribution.

        That is ``obs`` should be an array of non-negative integers from
        range ``[min(obs), max(obs)]``, such that each integer from the range
        occurs in ``obs`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(obs)
        if (len(symbols) == 1 or          # not enough data
            symbols.dtype.kind != 'i' or  # not an integer
            np.any(symbols < 0)):         # contains negative integers
            return False

        symbols.sort()
        return np.all(np.diff(symbols) <= 1)

    def fit(self, obs):
        if not self._check_input_symbols(obs):
            raise ValueError("expected a sample from "
                             "a Multinomial distribution.")

        return super(MultinomialHMM, self).fit(obs)


if __name__ == "__main__":
    mhmm = MultinomialHMM(n_components=2)
    obs = [[0, 1]*10]
    print obs
    mhmm.fit(obs)
    print "mhmm.trans_mat", mhmm.trans_mat
    print " mhmm.start_prob", mhmm.start_prob
    print "mhmm.emissionprob_", mhmm.emissionprob_
    print "mhmm.sequence", mhmm.decode([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,0])

    print '-----------------------------------------\n'
    mhmm2 = testHmm.MultinomialHMM(n_components=2, transmat=np.array([[0.5, 0.5],[0.5, 0.5]]), random_state=1)
    mhmm2.fit(obs)
    print "mhmm2.trans_mat", mhmm2._get_transmat()
    print " mhmm2.start_prob", mhmm2._get_startprob()
    print "mhmm2.emissionprob_", mhmm2._get_emissionprob()
    print mhmm2.decode(obs[0])