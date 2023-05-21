from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import librosa

# model = pickle.load(open('gmm_model.pkl', 'rb'))
application = Flask(__name__)


@application.route('/')
def home():
    return "hello world"


# @app.route('/speech', methods=['POST'])
# def speech():
#     f = request.files['file']
#     f.save(secure_filename(f.filename))
#
#     result = model.predict(f)


with open('gmm_model.pkl', 'rb') as file1:
    model = pickle.load(file1)


# Define predict_speaker function
@application.route('/speech', methods=['POST'])
def predict_speaker():
    # Load audio file
    f = request.files.get('file')
    signal, sr = librosa.load(f, sr=None)

    # Extract Mel-frequency cepstral coefficients (MFCCs)
    n_fft = 2048
    hop_length = int(sr * 0.01)
    n_mfcc = 20
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    # Use GMM to predict speaker labels for each frame
    posteriors = model.predict_proba(mfccs.T)
    speaker_labels = np.argmax(posteriors, axis=1)

    # Post-process speaker labels to remove short segments
    min_segment_len = 1  # seconds
    min_segment_frames = int(min_segment_len * sr / hop_length)
    speaker_labels_filtered = np.zeros_like(speaker_labels)
    for speaker_id in range(model.n_components):
        speaker_mask = speaker_labels == speaker_id
        speaker_segments = np.split(np.arange(len(speaker_labels))[speaker_mask],
                                    np.where(np.diff(np.arange(len(speaker_mask))[speaker_mask]) != 1)[0] + 1)
        for segment in speaker_segments:
            if len(segment) >= min_segment_frames:
                speaker_labels_filtered[segment] = speaker_id

    # Compute start and end times for each speaker segment
    segment_starts = np.where(np.diff(speaker_labels_filtered) != 0)[0] + 1
    segment_ends = np.append(segment_starts[1:], len(speaker_labels_filtered))
    speaker_ids = speaker_labels_filtered[segment_starts]

    segment_durations = (segment_ends - segment_starts) * hop_length / sr
    min_segment_len = 1  # seconds
    long_enough_mask = segment_durations >= min_segment_len
    segment_starts_long_enough = segment_starts[long_enough_mask]
    segment_ends_long_enough = segment_ends[long_enough_mask]
    speaker_ids_long_enough = speaker_ids[long_enough_mask]

    # Create list of dictionaries for speaker segments
    results = []
    for i in range(len(segment_starts_long_enough)):
        start_time = segment_starts_long_enough[i] * hop_length / sr
        end_time = segment_ends_long_enough[i] * hop_length / sr
        speaker_id = speaker_ids_long_enough[i]
        duration = segment_durations[long_enough_mask][i]
        results.append({'speaker': speaker_id, 'start_time': start_time, 'end_time': end_time, 'duration': duration})

    # Return results as JSON
    return jsonify(str(results))


if __name__ == '__main__':
    application.run(debug=True)
