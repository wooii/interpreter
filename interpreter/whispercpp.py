from pywhispercpp.model import Model, Segment
from time import time
from pathlib import Path
from typing import Union, Callable, List
import _pywhispercpp as pw
import numpy as np
import importlib.metadata
import logging
from interpreter import data_folder

__version__ = importlib.metadata.version('pywhispercpp')

logger = logging.getLogger(__name__)


class MySegment(Segment):
    """Add per-segment average token probability to Segment."""
    def __init__(self, t0: int, t1: int, text: str, probability: float = 0.0):
        """
        :param t0: start time
        :param t1: end time
        :param text: text
        :param probability: average token confidence
        """
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability

    def __str__(self):
        return f"t0={self.t0}, t1={self.t1}, text={self.text}, probability={self.probability}"


class WhisperCppModel(Model):
    """Wrapper around pywhispercpp Model with probability-aware segments."""
    @staticmethod
    def _get_segments(ctx, start: int, end: int) -> List[Segment]:
        """
        Helper function to get generated segments between `start` and `end`

        :param start: start index
        :param end: end index

        :return: list of segments
        """
        n = pw.whisper_full_n_segments(ctx)
        assert end <= n, f"{end} > {n}: `End` index must be less or equal than the total number of segments"
        res = []
        for i in range(start, end):
            t0 = pw.whisper_full_get_segment_t0(ctx, i)
            t1 = pw.whisper_full_get_segment_t1(ctx, i)
            bytes = pw.whisper_full_get_segment_text(ctx, i)
            text = bytes.decode('utf-8', errors='replace')
            n_tokens = pw.whisper_full_n_tokens(ctx, i)
            if n_tokens == 1:
                avg_prob = pw.whisper_full_get_token_p(ctx, i, 0)
            elif n_tokens > 1:
                probs = np.empty(n_tokens, dtype=np.float32)
                for j in range(n_tokens):
                    probs[j] = pw.whisper_full_get_token_p(ctx, i, j)
                avg_prob = probs.mean()
            else:
                avg_prob = 0.0
            res.append(MySegment(t0, t1, text.strip(), probability=np.float32(avg_prob)))
        return res

    def transcribe(self,
                   media: Union[str, np.ndarray],
                   n_processors: int = None,
                   new_segment_callback: Callable[[Segment], None] = None,
                   **params) -> List[Segment]:
        """
        Transcribes the media provided as input and returns list of `Segment` objects.
        Accepts a media_file path (audio/video) or a raw numpy array.

        :param media: Media file path or a numpy array
        :param n_processors: if not None, it will run the transcription on multiple processes
                             binding to whisper.cpp/whisper_full_parallel
                             > Split the input audio in chunks and process each chunk separately using whisper_full()
        :param new_segment_callback: callback function that will be called when a new segment is generated
        :param params: keyword arguments for different whisper.cpp parameters, see ::: constants.PARAMS_SCHEMA

        :return: List of transcription segments
        """
        if type(media) is np.ndarray:
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(media)
        # update params if any
        self._set_params(params)

        # setting up callback
        if new_segment_callback:
            WhisperCppModel._new_segment_callback = new_segment_callback
            pw.assign_new_segment_callback(self._params, WhisperCppModel.__call_new_segment_callback)

        # run inference
        start_time = time()
        logger.info("Transcribing ...")
        res = self._transcribe(audio, n_processors=n_processors)
        end_time = time()
        logger.info(f"Inference time: {end_time - start_time:.3f} s")
        return res

    def _transcribe(self, audio: np.ndarray, n_processors: int = None):
        """
        Private method to call the whisper.cpp/whisper_full function

        :param audio: numpy array of audio data
        :param n_processors: if not None, it will run whisper.cpp/whisper_full_parallel with n_processors
        :return:
        """

        if n_processors:
            pw.whisper_full_parallel(self._ctx, self._params, audio, audio.size, n_processors)
        else:
            pw.whisper_full(self._ctx, self._params, audio, audio.size)
        n = pw.whisper_full_n_segments(self._ctx)
        res = WhisperCppModel._get_segments(self._ctx, 0, n)
        return res

    @staticmethod
    def __call_new_segment_callback(ctx, n_new, user_data) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :param ctx: whisper.cpp ctx param
        :param n_new: whisper.cpp n_new param
        :param user_data: whisper.cpp user_data param
        :return: None
        """
        n = pw.whisper_full_n_segments(ctx)
        start = n - n_new
        res = WhisperCppModel._get_segments(ctx, start, n)
        for segment in res:
            WhisperCppModel._new_segment_callback(segment)


if __name__ == "__main__":
    model = WhisperCppModel("small",
                    token_timestamps=True,
                    max_len=1,
                    split_on_word=True,
                    print_progress=False,
                    )
    audio_path = data_folder / 'interpreter'/ "comparison_recording.wav"
    start = time()
    output =model.transcribe(str(audio_path),
                             temperature=0.0,
                             )
    end = time()
    print(end-start)


