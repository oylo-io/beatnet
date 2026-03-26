# BeatNet (Oylo Fork)

Fork of [mjhydri/BeatNet](https://github.com/mjhydri/BeatNet) — real-time beat, downbeat, and tempo tracking using a CRNN + particle filter.

## What changed in this fork

**Removed the madmom dependency entirely.** The original BeatNet depends on [madmom](https://github.com/CPJKU/madmom), which has abandoned Cython extensions incompatible with numpy>=2.0 and Python>=3.12, making BeatNet uninstallable on modern systems.

Also removed: `pyaudio`, `matplotlib`, `madmom`, `cython`, `numba` dependencies.

### Specific changes

- **Feature extraction** (`log_spect.py`): Replaced madmom's signal processing pipeline with pure numpy/scipy. Pre-computed logarithmic filterbank shipped as `.npy`.
- **State space** (`state_space.py`): Extracted `BarStateSpace`, `BarTransitionModel`, `TransitionModel`, `ObservationModel` from madmom source (pure Python/numpy, no Cython).
- **Particle filter** (`particle_filtering_cascade.py`): Updated imports to use extracted state space. Removed matplotlib plotting code.
- **BeatNet** (`BeatNet.py`): Removed pyaudio, matplotlib, and DBN (offline) mode. Added `feed()` streaming API for feeding PCM chunks directly.
- **Dependencies**: Only `numpy`, `scipy`, `librosa`, `torch`.

### Equivalence testing

Tested against original madmom-based pipeline on 30s audio samples:
- Feature extraction correlation: **0.9999**
- `feed()` beat detection: **43/43 beats matched** within 150ms

## Installation

```bash
pip install git+https://github.com/oylo-io/beatnet.git
```

Or for development:
```bash
git clone https://github.com/oylo-io/beatnet.git
pip install -e beatnet/
```

## Usage

### Streaming mode (feed PCM chunks)

```python
from BeatNet.BeatNet import BeatNet
import numpy as np

bn = BeatNet(model=1, mode='stream', inference_model='PF', plot=[], thread=False)

# Feed 22050 Hz mono float32 audio in chunks
chunk = np.random.randn(441).astype(np.float32)  # 20ms at 22050 Hz
result = bn.feed(chunk)
if result is not None:
    for beat_time, beat_type in result:
        # beat_type: 1 = downbeat, 2 = beat
        print(f"{'Downbeat' if beat_type == 1 else 'Beat'} at {beat_time:.3f}s")
```

### Process whole file

```python
from BeatNet.BeatNet import BeatNet

bn = BeatNet(model=1, mode='online', inference_model='PF', plot=[], thread=False)
beats = bn.process('audio.wav')
# beats: numpy array (num_beats, 2) — [time, beat_type]
```

## Credits

Original BeatNet by Mojtaba Heydari — [paper](https://arxiv.org/abs/2108.03576), [repo](https://github.com/mjhydri/BeatNet).

State space models based on Krebs, Böck & Widmer, "An Efficient State Space Model for Joint Tempo and Meter Tracking", ISMIR 2015 (originally implemented in madmom).

## Cite

```
@inproceedings{heydari2021beatnet,
  title={BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking},
  author={Heydari, Mojtaba and Cwitkowitz, Frank and Duan, Zhiyao},
  journal={22th International Society for Music Information Retrieval Conference, ISMIR},
  year={2021}
}
```
