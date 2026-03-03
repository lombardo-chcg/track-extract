# track-extract

A tool to segment extended live music performance recordings (e.g. concert sets captured on devices like the Zoom H-series) into individual, discrete song files.  It processes large `.wav` files and identifies track boundaries, making it easy to archive and share individual live tracks.

# install

Install `uv` first - [https://docs.astral.sh/uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

# use

```bash
# basic use
uv run main.py --input-file "ZOOM0001.WAV" --output-dir "/path/to/output"

# shorthand
uv run main.py -i "ZOOM0001.WAV" -o "/path/to/output"

# omit output arg to default to current dir
uv run main.py -i "ZOOM0001.WAV"
```

Modify `main.py` config params directly as needed (`SILENCE_THRESH`, `MIN_SILENCE_LEN_SEC`, `MIN_SONG_LEN_SEC`, etc.)