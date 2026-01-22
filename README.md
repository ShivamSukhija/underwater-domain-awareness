# underwater-domain-awareness

This project provides a complete pipeline for the classification and analysis of underwater acoustic signals. It includes tools for segmenting long audio recordings, extracting robust acoustic features (MFCCs, Mel-spectrograms), and training a Convolutional Neural Network (CNN) to identify different sound domains such as marine animals, vessels, and natural environments.

Features:

Multi-Format Support: Feature extraction scripts for .wav, .mp3, and .mp4 audio files.

Acoustic Feature Engineering: Extracts Mel-spectrograms, MFCCs (with Delta and Delta-Delta coefficients), Root Mean Square (RMS) energy, and Zero-Crossing Rate (ZCR).

Audio Segmentation: Automatically chunks long recordings into fixed-length segments (default 10s) with customizable overlap.

Deep Learning Model: Implements a CNNWithGAP (Convolutional Neural Network with Global Average Pooling) optimized for spectrogram-based classification.

End-to-End Notebooks: Ready-to-use Jupyter notebooks for feature generation, model training, and inference.

Acoustic Features Extracted:

For every 10-second audio segment, the pipeline generates:

Mel Spectrogram: Power-to-dB scaled spectrogram.

MFCCs: 20 Mel-frequency cepstral coefficients.

MFCC Deltas: First and second-order derivatives to capture temporal dynamics.

Temporal Stats: RMS energy and Zero-Crossing Rate.

Class Definitions:

The current implementation supports the following classes:
Anthropogenic

Marine Animal (e.g., Harp Seal)

Natural Sound

Vessel


For this project we curated a dataset of more than 10GB in size from the dataset which had audio files classified into these four classes.
