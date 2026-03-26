# State space and transition/observation models for beat tracking.
# Extracted from madmom (CPJKU/madmom, MIT license).
#
# Original sources:
#   madmom/features/beats_hmm.py — BeatStateSpace, BarStateSpace,
#       BarTransitionModel, exponential_transition
#   madmom/ml/hmm.pyx — TransitionModel, ObservationModel
#
# References:
#   Krebs, Böck & Widmer, "An Efficient State Space Model for Joint Tempo
#   and Meter Tracking", ISMIR 2015.

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import csr_matrix


class TransitionModel:
    """Sparse transition model (CSR format) for HMM-based tracking."""

    def __init__(self, states, pointers, probabilities):
        self.states = states
        self.pointers = pointers
        self.probabilities = probabilities

    @property
    def num_states(self):
        return len(self.pointers) - 1

    @property
    def num_transitions(self):
        return len(self.probabilities)

    @property
    def log_probabilities(self):
        return np.log(self.probabilities)

    @staticmethod
    def make_dense(states, pointers, probabilities):
        """Convert CSR sparse representation to dense (states, prev_states, probs)."""
        transitions = csr_matrix((np.array(probabilities),
                                  np.array(states), np.array(pointers)))
        states_out, prev_states = transitions.nonzero()
        return states_out, prev_states, probabilities

    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        """Convert dense (states, prev_states, probs) to CSR sparse representation."""
        states = np.asarray(states)
        prev_states = np.asarray(prev_states, dtype=int)
        probabilities = np.asarray(probabilities)
        num_states = max(prev_states) + 1
        transitions = csr_matrix((probabilities, (states, prev_states)),
                                 shape=(num_states, num_states))
        states_out = transitions.indices.astype(np.uint32)
        pointers = transitions.indptr.astype(np.uint32)
        probabilities_out = transitions.data.astype(float)
        return states_out, pointers, probabilities_out

    @classmethod
    def from_dense(cls, states, prev_states, probabilities):
        transitions = cls.make_sparse(states, prev_states, probabilities)
        return cls(*transitions)


class ObservationModel:
    """Observation model mapping HMM states to density columns via pointers."""

    def __init__(self, pointers):
        self.pointers = pointers

    def log_densities(self, observations):
        raise NotImplementedError('must be implemented by subclass')

    def densities(self, observations):
        return np.exp(self.log_densities(observations))


# --- State spaces ---

class BeatStateSpace:
    """State space for a single beat (phase × tempo grid)."""

    def __init__(self, min_interval, max_interval, num_intervals=None):
        intervals = np.arange(np.round(min_interval),
                              np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(np.log2(min_interval),
                                        np.log2(max_interval),
                                        num_log_intervals, base=2)
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1
        self.intervals = np.ascontiguousarray(intervals, dtype=np.int64)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(np.int64)
        self.last_states = np.cumsum(self.intervals) - 1
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.int64)
        idx = 0
        for i in self.intervals:
            self.state_positions[idx: idx + i] = np.linspace(0, 1, i,
                                                             endpoint=False)
            self.state_intervals[idx: idx + i] = i
            idx += i


class BarStateSpace:
    """State space for a bar (multiple beats stacked)."""

    def __init__(self, num_beats, min_interval, max_interval,
                 num_intervals=None):
        self.num_beats = int(num_beats)
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=np.int64)
        self.num_states = 0
        self.first_states = []
        self.last_states = []
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for b in range(self.num_beats):
            self.state_positions = np.hstack((self.state_positions,
                                              bss.state_positions + b))
            self.state_intervals = np.hstack((self.state_intervals,
                                              bss.state_intervals))
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            self.num_states += bss.num_states


# --- Transition distributions ---

def exponential_transition(from_intervals, to_intervals, transition_lambda,
                           threshold=np.spacing(1), norm=True):
    """Exponential tempo transition probability matrix."""
    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals),
                                        len(to_intervals)))))
    ratio = (to_intervals.astype(np.float64) /
             from_intervals.astype(np.float64)[:, np.newaxis])
    prob = np.exp(-transition_lambda * abs(ratio - 1.))
    prob[prob <= threshold] = 0
    if norm:
        prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob


class BarTransitionModel(TransitionModel):
    """Transition model for bar tracking (exponential tempo transitions at beat boundaries)."""

    def __init__(self, state_space, transition_lambda):
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_beats
        self.state_space = state_space
        self.transition_lambda = transition_lambda
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float64)
        for beat in range(state_space.num_beats):
            to_states = state_space.first_states[beat]
            from_states = state_space.last_states[beat - 1]
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int, to_int,
                                          transition_lambda[beat])
            from_prob, to_prob = np.nonzero(prob)
            states = np.hstack((states, to_states[to_prob]))
            prev_states = np.hstack((prev_states, from_states[from_prob]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))
        transitions = self.make_sparse(states, prev_states, probabilities)
        super().__init__(*transitions)
