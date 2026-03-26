# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# Modified: removed madmom/matplotlib dependencies

import numpy as np
from numpy.random import default_rng
rng = default_rng()
from BeatNet.state_space import BarStateSpace, BarTransitionModel, TransitionModel, ObservationModel

class BDObservationModel(ObservationModel):
    """
    Observation model for beat and downbeat tracking with particle filtering.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : str
        Based on the first character of this parameter, each (down-)beat period gets split into (down-)beat states
        "B" stands for border model which classifies 1/(observation lambda) fraction of states as downbeat states and
        the rest as the beat states (if it is used for downbeat tracking state space) or the same fraction of states
        as beat states and the rest as the none beat states (if it is used for beat tracking state space).
        "N" model assigns a constant number of the beginning states as downbeat states and the rest as beat states
         or beginning states as beat and the rest as none-beat states
        "G" model is a smooth Gaussian transition (soft border) between downbeat/beat or beat/none-beat states

    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':
            observation_lambda = int(observation_lambda[1:])
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            pointers[state_space.state_positions < border] = 2
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'N':
            observation_lambda = int(observation_lambda[1:])
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                pointers[border[0]] = 2
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'G':
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            super(BDObservationModel, self).__init__(pointers)


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)


#   assigning beat vs non-beat weights
def beat_densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(observation_model.pointers == 2)] = observations
        new_obs[np.argwhere(observation_model.pointers == 0)] = 0.03
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

#   assigning downbeat vs beat weights
def down_densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations[1]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = observations[0]
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

#   assigning downbeat vs beat weights - second model
def down_densities2(observations, beats_per_bar):
    new_obs = np.zeros(beats_per_bar, float)
    new_obs[0] = observations[1]  # downbeat activation
    new_obs[1:] = observations[0]  # beat activation
    return new_obs

#   Inference initialization
class particle_filter_cascade:
    np.random.seed(1)
    PARTICLE_SIZE = 1500   #  1500
    DOWN_PARTICLE_SIZE = 250  # 250
    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = 300
    LAMBDA_B = 60  # beat transition lambda
    LAMBDA_D = 0.1  # downbeat transition lambda
    OBSERVATION_LAMBDA_B = "B56"  # beat observation lambda
    OBSERVATION_LAMBDA_D = "B56"  # downbeat observation lambda
    fps = 50
    T = 1 / fps
    MIN_BEAT_PER_BAR = 2
    MAX_BEAT_PER_BAR = 4
    OFFSET = 0 # The point of time after which the inference model starts to work. Can be zero!
    IG_THRESHOLD = 0.4  # Information Gate threshold

    def __init__(self, beats_per_bar=[], particle_size=PARTICLE_SIZE, down_particle_size=DOWN_PARTICLE_SIZE,
                 min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, min_beats_per_bar=MIN_BEAT_PER_BAR,
                 max_beats_per_bar=MAX_BEAT_PER_BAR, offset=OFFSET, ig_threshold=IG_THRESHOLD, lambda_b=LAMBDA_B,
                 lambda_d=LAMBDA_D, observation_lambda_b=OBSERVATION_LAMBDA_B, observation_lambda_d=OBSERVATION_LAMBDA_D,
                 fps=None, plot=False, mode=None, **kwargs):
        self.particle_size = particle_size
        self.down_particle_size = down_particle_size
        self.particle_filter = []
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.Lambda_b = lambda_b
        self.Lambda_d = lambda_d
        self.observation_lambda_b = observation_lambda_b
        self.observation_lambda_d = observation_lambda_d
        self.plot = plot
        self.min_beats_per_bar = min_beats_per_bar
        self.max_beats_per_bar = max_beats_per_bar
        self.offset = offset
        self.ig_threshold = ig_threshold
        self.mode = mode
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BarStateSpace(1, min_interval, max_interval, num_tempi)    # beat tracking state space
        if beats_per_bar:   # if the number of beats per bar is given
            self.st2 = BarStateSpace(1, min(self.beats_per_bar ), max(self.beats_per_bar),
                                max(self.beats_per_bar ) - min(self.beats_per_bar) + 1)   # downbeat tracking state space
        else:   # if the number of beats per bar is not given
            self.st2 = BarStateSpace(1, self.min_beats_per_bar, self.max_beats_per_bar, self.max_beats_per_bar - self.min_beats_per_bar + 1)  # downbeat tracking state space
        tm = BarTransitionModel(self.st, self.Lambda_b)
        self.tm = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))   # beat transition model
        self.om = BDObservationModel(self.st, self.observation_lambda_b)   # beat observation model
        self.st.last_states = list(np.concatenate(self.st.last_states).flat)    # beat last states
        self.om2 = BDObservationModel(self.st2, self.observation_lambda_d)  # downbeat observation model
        self.tm2 = np.zeros((len(self.st2.first_states[0]), len(self.st2.first_states[0])))  # downbeat transition model
        for i in range(len(self.st2.first_states[0])):
            for j in range(len(self.st2.first_states[0])):
                if i == j:
                    self.tm2[i, j] = 1 - self.Lambda_d
                else:
                    self.tm2[i, j] = self.Lambda_d / (len(self.st2.first_states[0]) - 1)
        self.T = 1 / self.fps
        self.counter = -1
        self.path = np.zeros((1, 2), dtype=float)

        #   particles initialization
        self.particles = np.sort(np.random.choice(np.arange(0, self.st.num_states - 1), self.particle_size, replace=True))
        self.down_particles = np.sort(np.random.choice(np.arange(0, self.st2.num_states - 1), self.down_particle_size, replace=True))
        self.beat = np.squeeze(self.st.first_states)

    def process(self, activations):
        """
        Running Particle filtering over the given activation function to infer beats/downbeats.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats, downbeats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        #   applying the offset and information gate thresholds
        activations = activations[int(self.offset / self.T):]
        if np.shape(activations)==(2,):
            activations = np.reshape(activations, (-1, 2))
        both_activations = activations.copy()
        activations = np.max(activations, axis=1)
        activations[activations < self.ig_threshold] = 0.03
        self.activations = activations
        self.both_activations = both_activations

        for i in range(len(activations)):  # loop through the provided frame/s to infer beats/downbeats
            self.counter += 1
            gathering = int(np.median(self.particles))   # calculating beat particles clutter
            # checking if the clutter is within the beat interval
            if ((gathering - self.beat[self.st.state_intervals[self.beat] == self.st.state_intervals[gathering]]) < (
                    int(.07 / self.T)) + 1).any() and (self.offset + self.counter * self.T) - self.path[-1][0] > .4 * self.T * \
                    self.st.state_intervals[gathering]:

                # downbeat particles motion
                last1 = self.down_particles[np.in1d(self.down_particles, self.st2.last_states)]
                state1 = self.down_particles[~np.in1d(self.down_particles, self.st2.last_states)] + 1
                for j in range(len(last1)):
                    arg1 = np.argwhere(self.st2.last_states[0] == last1[j])[0][0]
                    nn = np.random.choice(self.st2.first_states[0], 1, p=(np.squeeze(self.tm2[arg1])))
                    state1 = np.append(state1, nn)
                self.down_particles = state1

                # downbeat particles correction
                if both_activations[i][1]>0.7:
                    self.down_particles = np.append(self.down_particles,np.array([self.st2.first_states]))
                obs2 = down_densities(both_activations[i], self.om2, self.st2)
                self.down_particles = universal_resample(self.down_particles, obs2[self.down_particles])
                if both_activations[i][1]>0.7:
                    self.down_particles = np.delete(self.down_particles, np.random.choice(self.down_particle_size, len(self.st2.first_states), replace=False))
                m = np.bincount(self.down_particles)
                self.down_max = np.argmax(m)  # calculating downbeat particles clutter

                # beat vs downbeat distinguishment
                if self.down_max in self.st2.first_states[0] and self.path[-1][1] !=1 and both_activations[i][1]>0.4:
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 1]], axis=0)
                    if self.mode == 'stream' or self.mode == 'realtime':
                        print("*beat!")
                elif (activations[i]>0.4) :
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 2]], axis=0)
                    if self.mode == 'stream' or self.mode == 'realtime':
                        print("beat!")

            # beat particles motion
            last = self.particles[np.in1d(self.particles, self.st.last_states)]
            state = self.particles[~np.in1d(self.particles, self.st.last_states)] + 1
            for j in range(len(last)):
                args = np.argwhere(self.tm[1] == last[j])
                probs = self.tm[2][args]
                nn = np.random.choice(np.squeeze(self.tm[0][args]), 1, p=(np.squeeze(probs)))
                state = np.append(state, nn)
            self.particles = state

            # beat particles correction
            obs = beat_densities(activations[i], self.om, self.st)
            if activations[i] > 0.1:  # resampling is done only when there is a meaningful activation
                if activations[i] > 0.8:
                    self.particles = np.append(self.particles,np.array([self.st.first_states[0][np.arange(np.random.randint(4),len(self.st.first_states[0]),6)]]))
                self.particles = universal_resample(self.particles, obs[self.particles], )  # beat correction
                if activations[i] > 0.8:
                    np.delete(self.particles, np.random.choice(self.particle_size, len(self.st.first_states), replace=False))
        return self.path[1:]


def universal_resample(particles, weights):
    J = len(particles)
    weights = weights / sum(weights)
    cumsum_weights = np.cumsum(weights)
    r = np.random.uniform(0, 1 / J, J)
    U = r + np.arange(J) * (1 / J)
    new_particles = particles[np.searchsorted(cumsum_weights, U)]
    return new_particles
