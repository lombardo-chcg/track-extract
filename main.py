import argparse
import librosa
import numpy as np
import soundfile as sf
import os

# === CONFIGURATION - customize these values for your use case===
SILENCE_THRESH = 0.02       # Volume threshold (RMS)
MIN_SILENCE_LEN_SEC = 2.0   # Minimum silence length
MIN_SONG_LEN_SEC = 60.0     # Ignore chunks shorter than this
MERGE_GAP_SEC = 5.0         # Merge gaps smaller than this

def parse_args():
    parser = argparse.ArgumentParser(description="Extract songs from a stereo WAV file based on silence detection.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to input WAV file')
    parser.add_argument('--output-dir', type=str, default='songs_st', help='Directory to save extracted songs')
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    print(f"processing {input_file}, saving tracks to ${output_dir}")

    # === LOAD AUDIO IN STEREO ===
    y_stereo, sr = sf.read(input_file, always_2d=True)
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
    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(segments):
        duration = end - start
        if duration < MIN_SONG_LEN_SEC:
            continue
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y_stereo[start_sample:end_sample]
        out_path = os.path.join(output_dir, f"song_{i+1:02}.wav")
        sf.write(out_path, chunk, sr)
        print(f"Exported {out_path} [{duration:.2f} sec]")

    print("Done.")

if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
