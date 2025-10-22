import librosa
import numpy as np
import soundfile as sf
import os

# === CONFIGURATION - customize these values for your use case===
INPUT_FILE = "ZOOM0002.WAV"
OUTPUT_DIR = "songs_st"
SILENCE_THRESH = 0.02       # Volume threshold (RMS)
MIN_SILENCE_LEN_SEC = 2.0   # Minimum silence length
MIN_SONG_LEN_SEC = 60.0     # Ignore chunks shorter than this
MERGE_GAP_SEC = 5.0         # Merge gaps smaller than this

# === LOAD AUDIO IN STEREO ===
y_stereo, sr = sf.read(INPUT_FILE, always_2d=True)
y_mono = y_stereo.mean(axis=1)  # Mix down for RMS-based silence detection

# === CALCULATE RMS OVER MONO ===
frame_length = int(sr * 0.1)  # 100 ms
hop_length = frame_length
rms = librosa.feature.rms(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]

# === FIND NON-SILENT SEGMENTS ===
frames = np.where(rms > SILENCE_THRESH)[0]
if len(frames) == 0:
    raise ValueError("No non-silent regions detected!")

times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
segments = []
start = times[0]
prev = times[0]

for t in times[1:]:
    if t - prev > MERGE_GAP_SEC:
        segments.append((start, prev))
        start = t
    prev = t
segments.append((start, prev))

# === EXPORT SONGS FROM ORIGINAL STEREO AUDIO ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, (start, end) in enumerate(segments):
    duration = end - start
    if duration < MIN_SONG_LEN_SEC:
        continue
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = y_stereo[start_sample:end_sample]
    out_path = os.path.join(OUTPUT_DIR, f"song_{i+1:02}.wav")
    sf.write(out_path, chunk, sr)
    print(f"Exported {out_path} [{duration:.2f} sec]")

print("Done.")
