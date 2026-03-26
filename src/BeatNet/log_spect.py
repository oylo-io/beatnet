# Audio feature extraction for BeatNet
# Replaces original madmom-based pipeline with pure numpy/scipy
#
# Original chain: SignalProcessor → FramedSignalProcessor → STFT →
#   FilteredSpectrogram(24 bands/octave, 30-17000Hz, LogarithmicFilterbank) →
#   Log(mul=1, add=1) → SpectrogramDifference(ratio=0.5, positive_diffs=True, hstack)

import os
import numpy as np
from scipy.fft import rfft

from BeatNet.common import FeatureModule

A4 = 440.0  # reference tuning frequency


def _log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """Logarithmically spaced frequencies (same as madmom.audio.filters.log_frequencies)."""
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    frequencies = fref * 2. ** (np.arange(left, right) / float(bands_per_octave))
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    return frequencies


def _frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """Map frequencies to closest FFT bins (same as madmom.audio.filters.frequencies2bins)."""
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    if unique_bins:
        indices = np.unique(indices)
    return indices


def _triangular_filters(bins, num_fft_bins, norm=True):
    """Create overlapping triangular filters from bin positions.

    Reimplements madmom.audio.filters.TriangularFilter.filters with overlap=True.
    """
    num_filters = len(bins) - 2  # bins includes left edge, center, right edge pairs
    filterbank = np.zeros((num_fft_bins, num_filters))
    for i in range(num_filters):
        start = bins[i]
        center = bins[i + 1]
        stop = bins[i + 2]
        # rising slope
        if center > start:
            rising = np.arange(start, center)
            filterbank[rising, i] = (rising - start) / (center - start)
        # peak
        filterbank[center, i] = 1.0
        # falling slope
        if stop > center:
            falling = np.arange(center + 1, stop + 1)
            if len(falling) > 0:
                filterbank[falling[falling < num_fft_bins], i] = \
                    (stop - falling[falling < num_fft_bins]) / (stop - center)
        # normalize to area 1
        if norm:
            area = filterbank[:, i].sum()
            if area > 0:
                filterbank[:, i] /= area
    return filterbank


def _build_logarithmic_filterbank(n_fft, sample_rate, num_bands=24,
                                   fmin=30, fmax=17000, fref=A4):
    """Build a logarithmic filterbank matrix matching madmom's LogarithmicFilterbank."""
    # Exclude Nyquist bin to match madmom (include_nyquist=False)
    bin_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[:-1]
    # Get logarithmically spaced center frequencies
    center_freqs = _log_frequencies(num_bands, fmin, fmax, fref)
    # Map to FFT bins (unique to avoid duplicates at low freq)
    bins = _frequencies2bins(center_freqs, bin_frequencies, unique_bins=True)
    # The bins array already includes edge positions (first and last bins act as
    # edges for the first/last filters). madmom uses overlap=True, meaning
    # adjacent filters share edges: filter_i uses bins[i], bins[i+1], bins[i+2].
    # 138 bins → 136 filters.
    filterbank = _triangular_filters(bins, len(bin_frequencies), norm=True)
    return filterbank


class LOG_SPECT(FeatureModule):
    def __init__(self, num_channels=1, sample_rate=22050, win_length=2048,
                 hop_size=512, n_bands=None, mode='online'):
        if n_bands is None:
            n_bands = [12]
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        self.win_length = win_length
        self.num_bands_per_octave = n_bands[0]
        self.mode = mode
        self._num_frames = 4 if mode in ('stream', 'realtime') else None

        # Load pre-computed filterbank if available, otherwise build it
        script_dir = os.path.dirname(__file__)
        fb_path = os.path.join(script_dir, 'models',
                               f'filterbank_{self.num_bands_per_octave}bpo_30_17000.npy')
        if os.path.exists(fb_path):
            self._filterbank = np.load(fb_path)
        else:
            self._filterbank = _build_logarithmic_filterbank(
                n_fft=win_length, sample_rate=sample_rate,
                num_bands=self.num_bands_per_octave, fmin=30, fmax=17000,
            )
        self._n_bands = self._filterbank.shape[1]
        self._window = np.hanning(win_length).astype(np.float32)

    def process_audio(self, audio):
        """Extract log-filtered spectrogram + positive half-wave rectified diff.

        Parameters
        ----------
        audio : np.ndarray, shape (num_samples,)
            Mono audio at self.sample_rate.

        Returns
        -------
        feats : np.ndarray, shape (2 * n_bands, num_frames)
            Stacked [log_spec; positive_diff] features.
        """
        # Frame the signal with centered framing (madmom pads win_length//2 at start)
        n_samples = len(audio)
        origin = self.win_length // 2
        # Pre-pad at start for centered framing
        audio = np.concatenate([np.zeros(origin, dtype=audio.dtype), audio])
        n_padded = len(audio)
        n_frames = int(np.ceil(n_samples / self.hop_length))
        if n_frames <= 0:
            return np.zeros((2 * self._n_bands, 0))

        # Pad at end so all frames fit
        pad_length = (n_frames - 1) * self.hop_length + self.win_length - n_padded
        if pad_length > 0:
            audio = np.concatenate([audio, np.zeros(pad_length, dtype=audio.dtype)])

        # Create frames using stride tricks
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, self.win_length),
            strides=(audio.strides[0] * self.hop_length, audio.strides[0]),
        ).copy()

        # Apply window
        frames *= self._window

        # STFT (magnitude), exclude Nyquist bin to match madmom
        stft = np.abs(rfft(frames, n=self.win_length, axis=1))[:, :-1]

        # Apply logarithmic filterbank
        filtered = stft @ self._filterbank  # (n_frames, n_bands)

        # Log compression: log10(1 * x + 1) — madmom defaults to log10, not ln
        log_spec = np.log10(filtered + 1)

        # Spectral difference: diff[t] = max(spec[t] - spec[t-1], 0)
        # (positive half-wave rectified, diff_frames=1)
        diff = np.zeros_like(log_spec)
        diff[1:] = log_spec[1:] - log_spec[:-1]
        np.maximum(diff, 0, out=diff)

        # Stack and transpose: (n_frames, 2*n_bands) -> (2*n_bands, n_frames)
        feats = np.hstack([log_spec, diff]).T

        # For streaming/realtime mode, only keep last num_frames
        if self._num_frames is not None and feats.shape[1] > self._num_frames:
            feats = feats[:, -self._num_frames:]

        return feats
