class ShotInfo(object):
    def __init__(self, data_root, target_fpath, audio_fpath, first_frame, last_frame, fps, audio_offset):
        self.data_root = data_root # prefix for fpath
        self.target_fpath = target_fpath # megashot
        self.audio_fpath = audio_fpath # megashot
        self.first_frame = first_frame # within megashot
        self.last_frame = last_frame # within megashot
        self.fps = fps
        self.audio_offset = audio_offset

    def __repr__(self):
        return 'Shot({}, [{}, {}])'.format(self.audio_fpath, self.first_frame, self.last_frame)

def read_shot_list(ds_info):
    shots = []
    for item in ds_info['shots']:
        shots.append(ShotInfo(
            data_root=ds_info['shots_root'],
            target_fpath=item[0][:-4]+'.xml', # FIXME must be general for all types of target GT (get path from ds_info)
            audio_fpath=item[0],
            first_frame=item[2],
            last_frame=item[3],
            fps=item[1],
            audio_offset=item[4],
        ))
    return shots

import numpy as np
def extract_random_segments(raw_speech_array, vertice, audio, sample_rate=16000, fps=30, min_frame=30, max_frame=150):
    """
    Extracts random segments from raw_speech_array, vertice, and audio tensors.
    """
    num_frames = np.random.randint(min_frame, max_frame + 1)
    num_frames = min(num_frames, vertice.shape[1])
    duration_seconds = num_frames / fps

    num_samples = int(duration_seconds * sample_rate)
    max_start_frame = vertice.shape[1] - num_frames
    start_frame = np.random.randint(0, max_start_frame + 1)
    start_sample = max(int(start_frame * (sample_rate / fps)), 0)

    raw_speech_segment = raw_speech_array[:, start_sample:start_sample + num_samples]
    audio_segment = audio[:, start_sample:start_sample + num_samples]
    vertice_segment = vertice[:, start_frame:start_frame + num_frames, :]

    return raw_speech_segment, vertice_segment, audio_segment


def normalize_data(data, mean = None, std = None):
    if mean is None and std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-8)
    return normalized_data, mean, std
