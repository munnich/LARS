# Label Audio Recording Segments

## Installing requirements

```
pip install -Ur src/requirements.txt
```

## Running

```
python src/lars.py
```

## Usage

1. Open an audio file, e.g. with the Ctrl+O shortcut.
2. Adjust frame length and overlap if necessary. 
3. In the bottom right field, enter your label and press Enter to proceed to the next frame.
4. Once you've finished labeling the entire file, a prompt will appear asking you to save to a CSV file.

If you've already split the audio file into frames, you can load these as a CSV file via the option in the File menu.

### Additional scripts

* `syllables.jl` contains a simple implementation of the syllable detection from J. Xu, W. Liao and T. Inoue, "Speech Speed Awareness System for a Non-native Speaker," 2016 International Conference on Collaboration Technologies and Systems (CTS), Orlando, FL, USA, 2016, pp. 43-50, doi: 10.1109/CTS.2016.0027 with CSV outputting. This is WIP and currently doesn't provide the desired results.
* `syllable_detection.py` contains a Python port of the above.
* `noisegatesegments.jl` use a noise gate to detect segments
