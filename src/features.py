import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
def extract_segments_and_log(file_path, out_dir, label, class_id,
                             sr=48000, segment_length=10, hop_length=10,
                             keep_short=True):
    """
    Segment audio into fixed-length chunks and extract features.
    Each segment saved as .npz, with metadata returned for DataFrame.

    Args:
        file_path (str): input .wav file
        out_dir (str): directory to save feature .npz files
        label (str): class label name
        class_id (int): numeric class ID
        sr (int): resample rate (default=48kHz)
        segment_length (int): chunk length in seconds
        hop_length (int): hop between chunks in seconds
        keep_short (bool): if True, pad clips shorter than segment_length

    Returns:
        rows (list[dict]): metadata rows for each segment
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    total_dur = librosa.get_duration(y=y, sr=sr)

    seg_samples = segment_length * sr
    hop_samples = hop_length * sr

    rows = []
    if len(y) < seg_samples:
        if not keep_short:
            return rows  # skip too short
        # pad short clips to segment_length
        y = librosa.util.fix_length(y, size=seg_samples)
        segments = [(0, y)]
    else:
        segments = []
        for start in range(0, len(y) - seg_samples + 1, hop_samples):
            end = start + seg_samples
            segments.append((start, y[start:end]))


    # Process each segment
    for i, (start, y_seg) in enumerate(segments):
        # Extract features
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        rms = librosa.feature.rms(y=y_seg)
        zcr = librosa.feature.zero_crossing_rate(y_seg)


        base_name = os.path.basename(file_path).replace(".wav", f"_{i:03d}.npz")
        feature_path = os.path.join(out_dir, base_name)
        np.savez(feature_path,
                 mel=mel_db, mfcc=mfcc, mfcc_delta=mfcc_delta,
                 mfcc_delta2=mfcc_delta2, rms=rms, zcr=zcr,
                 label=label, class_id=class_id,
                 original_path=file_path,
                 segment_start=start/sr, segment_dur=segment_length,
                 full_duration=total_dur)


        rows.append({
            "feature_path": feature_path,
            "class_id": class_id,
            "class_name": label,
            "original_path": file_path,
            "duration": total_dur,
            "segment_start": start/sr,
            "segment_dur": segment_length
        })
    return rows
def extract_segments_and_logmp3(file_path, out_dir, label, class_id,
                             sr=48000, segment_length=10, hop_length=10,
                             keep_short=True):
    """
    Segment audio into fixed-length chunks and extract features.
    Each segment saved as .npz, with metadata returned for DataFrame.

    Args:
        file_path (str): input .wav file
        out_dir (str): directory to save feature .npz files
        label (str): class label name
        class_id (int): numeric class ID
        sr (int): resample rate (default=48kHz)
        segment_length (int): chunk length in seconds
        hop_length (int): hop between chunks in seconds
        keep_short (bool): if True, pad clips shorter than segment_length

    Returns:
        rows (list[dict]): metadata rows for each segment
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    total_dur = librosa.get_duration(y=y, sr=sr)

    seg_samples = segment_length * sr
    hop_samples = hop_length * sr

    rows = []
    if len(y) < seg_samples:
        if not keep_short:
            return rows  # skip too short
        # pad short clips to segment_length
        y = librosa.util.fix_length(y, size=seg_samples)
        segments = [(0, y)]
    else:
        segments = []
        for start in range(0, len(y) - seg_samples + 1, hop_samples):
            end = start + seg_samples
            segments.append((start, y[start:end]))


    # Process each segment
    for i, (start, y_seg) in enumerate(segments):
        # Extract features
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        rms = librosa.feature.rms(y=y_seg)
        zcr = librosa.feature.zero_crossing_rate(y_seg)


        base_name = os.path.basename(file_path).replace(".mp3", f"_{i:03d}.npz")
        feature_path = os.path.join(out_dir, base_name)
        np.savez(feature_path,
                 mel=mel_db, mfcc=mfcc, mfcc_delta=mfcc_delta,
                 mfcc_delta2=mfcc_delta2, rms=rms, zcr=zcr,
                 label=label, class_id=class_id,
                 original_path=file_path,
                 segment_start=start/sr, segment_dur=segment_length,
                 full_duration=total_dur)


        rows.append({
            "feature_path": feature_path,
            "class_id": class_id,
            "class_name": label,
            "original_path": file_path,
            "duration": total_dur,
            "segment_start": start/sr,
            "segment_dur": segment_length
        })
    return rows
def extract_segments_and_logmp4(file_path, out_dir, label, class_id,
                             sr=48000, segment_length=10, hop_length=10,
                             keep_short=True):
    """
    Segment audio into fixed-length chunks and extract features.
    Each segment saved as .npz, with metadata returned for DataFrame.

    Args:
        file_path (str): input .wav file
        out_dir (str): directory to save feature .npz files
        label (str): class label name
        class_id (int): numeric class ID
        sr (int): resample rate (default=48kHz)
        segment_length (int): chunk length in seconds
        hop_length (int): hop between chunks in seconds
        keep_short (bool): if True, pad clips shorter than segment_length

    Returns:
        rows (list[dict]): metadata rows for each segment
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    total_dur = librosa.get_duration(y=y, sr=sr)

    seg_samples = segment_length * sr
    hop_samples = hop_length * sr

    rows = []
    if len(y) < seg_samples:
        if not keep_short:
            return rows  # skip too short
        # pad short clips to segment_length
        y = librosa.util.fix_length(y, size=seg_samples)
        segments = [(0, y)]
    else:
        segments = []
        for start in range(0, len(y) - seg_samples + 1, hop_samples):
            end = start + seg_samples
            segments.append((start, y[start:end]))


    # Process each segment
    for i, (start, y_seg) in enumerate(segments):
        # Extract features
        mel = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        rms = librosa.feature.rms(y=y_seg)
        zcr = librosa.feature.zero_crossing_rate(y_seg)


        base_name = os.path.basename(file_path).replace(".mp4", f"_{i:03d}.npz")
        feature_path = os.path.join(out_dir, base_name)
        np.savez(feature_path,
                 mel=mel_db, mfcc=mfcc, mfcc_delta=mfcc_delta,
                 mfcc_delta2=mfcc_delta2, rms=rms, zcr=zcr,
                 label=label, class_id=class_id,
                 original_path=file_path,
                 segment_start=start/sr, segment_dur=segment_length,
                 full_duration=total_dur)


        rows.append({
            "feature_path": feature_path,
            "class_id": class_id,
            "class_name": label,
            "original_path": file_path,
            "duration": total_dur,
            "segment_start": start/sr,
            "segment_dur": segment_length
        })
    return rows
