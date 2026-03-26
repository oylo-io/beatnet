# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# Modified: removed madmom/pyaudio/matplotlib dependencies, added feed() API

import os
import torch
import numpy as np
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.log_spect import LOG_SPECT
import librosa
from BeatNet.model import BDA


class BeatNet:

    '''
    The main BeatNet handler class including different trained models,
    different modes for extracting the activation and causal inferences.

    Parameters
    ----------
    model: int in [1,3]
        Which pre-trained CRNN model to use.
    mode: str
        'stream' — accepts PCM chunks via feed().
        'realtime' — reads audio file chunk by chunk.
        'online' — reads whole audio, uses PF inference.
    inference_model: str
        'PF' — Particle Filter (causal, for real-time).
    plot: list
        Unused, kept for API compatibility.
    thread: bool
        Unused, kept for API compatibility.
    device: str
        'cpu' or 'cuda:N'.
    '''

    def __init__(self, model, mode='online', inference_model='PF', plot=[],
                 thread=False, device='cpu'):
        if inference_model != 'PF':
            raise RuntimeError('Only PF inference is supported (madmom DBN removed)')
        self.mode = mode
        self.inference_model = inference_model
        self.device = device
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)  # 441
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)  # 1411
        self.proc = LOG_SPECT(
            sample_rate=self.log_spec_sample_rate,
            win_length=self.log_spec_win_length,
            hop_size=self.log_spec_hop_length,
            n_bands=[24],
            mode=self.mode,
        )
        self.estimator = particle_filter_cascade(
            beats_per_bar=[], fps=50, plot=[], mode=self.mode,
        )
        script_dir = os.path.dirname(__file__)
        self.model = BDA(272, 150, 2, self.device)
        if model == 1:
            weights_path = os.path.join(script_dir, 'models/model_1_weights.pt')
        elif model == 2:
            weights_path = os.path.join(script_dir, 'models/model_2_weights.pt')
        elif model == 3:
            weights_path = os.path.join(script_dir, 'models/model_3_weights.pt')
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True),
            strict=False,
        )
        self.model.eval()

        # Streaming state
        self.stream_window = np.zeros(
            self.log_spec_win_length + 2 * self.log_spec_hop_length,
            dtype=np.float32,
        )
        self.counter = 0

    def feed(self, pcm_chunk):
        """Feed a chunk of audio and return new beat detections.

        Parameters
        ----------
        pcm_chunk : np.ndarray, shape (N,), dtype float32
            Mono audio at self.sample_rate (22050 Hz).

        Returns
        -------
        new_beats : np.ndarray or None
            Array of shape (M, 2) with [time, beat_number] for each
            newly detected beat, or None if no new beats.
        """
        # Accumulate audio into sliding window
        self.stream_window = np.append(
            self.stream_window[len(pcm_chunk):], pcm_chunk,
        )

        prev_len = len(self.estimator.path)

        with torch.no_grad():
            if self.counter < 5:
                pred = np.zeros([1, 2])
            else:
                feats = self.proc.process_audio(self.stream_window)[:, -1]
                feats = torch.from_numpy(feats).float()
                feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(feats)[0]
                pred = self.model.final_pred(pred)
                pred = pred.cpu().detach().numpy()
                pred = np.transpose(pred[:2, :])

        self.estimator.process(pred)
        self.counter += 1

        new_len = len(self.estimator.path)
        if new_len > prev_len:
            return self.estimator.path[prev_len:]
        return None

    def process(self, audio_path=None):
        """Process audio file or array.

        Parameters
        ----------
        audio_path : str or np.ndarray
            Path to audio file, or audio array.

        Returns
        -------
        beats : np.ndarray, shape (num_beats, 2)
            Detected beat positions and beat numbers.
        """
        if self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if isinstance(audio_path, str):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            elif len(np.shape(audio_path)) > 1:
                audio = np.mean(audio_path, axis=1)
            else:
                audio = audio_path
            self.audio = audio
            while self.completed == 0:
                self._activation_extractor_realtime()
                output = self.estimator.process(self.pred)
                self.counter += 1
            return output

        elif self.mode == "online":
            if isinstance(audio_path, str):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            elif len(np.shape(audio_path)) > 1:
                audio = np.mean(audio_path, axis=1)
            else:
                audio = audio_path
            preds = self._activation_extractor_online(audio)
            output = self.estimator.process(preds)
            return output

        else:
            raise RuntimeError(f'process() not supported for mode={self.mode}. '
                               f'Use feed() for streaming.')

    def _activation_extractor_realtime(self):
        with torch.no_grad():
            if self.counter < (round(len(self.audio) / self.log_spec_hop_length)):
                if self.counter < 2:
                    self.pred = np.zeros([1, 2])
                else:
                    start = self.log_spec_hop_length * (self.counter - 2)
                    end = self.log_spec_hop_length * self.counter + self.log_spec_win_length
                    feats = self.proc.process_audio(self.audio[start:end])[:, -1]
                    feats = torch.from_numpy(feats).float()
                    feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                    pred = self.model(feats)[0]
                    pred = self.model.final_pred(pred)
                    pred = pred.cpu().detach().numpy()
                    self.pred = np.transpose(pred[:2, :])
            else:
                self.completed = 1

    def _activation_extractor_online(self, audio):
        with torch.no_grad():
            feats = self.proc.process_audio(audio)
            feats = torch.from_numpy(feats).float()
            feats = feats.unsqueeze(0).to(self.device)
            preds = self.model(feats)[0]
            preds = self.model.final_pred(preds)
            preds = preds.cpu().detach().numpy()
            preds = np.transpose(preds[:2, :])
        return preds
