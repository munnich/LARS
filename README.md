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

To correct a label, press Backspace or click the back button.

### Learning

Learning is done via a simple K-D tree. If turned on, every label is saved as a vector in the tree. For every new frame started, a noise gate is applied and the distance to the nearest previous frame is found. If this distance is below another threshold, the label is automatically suggested and the user only needs to press enter to apply it. These two thresholds for the noise gate and distance can be set after learning is turned on.

### Additional scripts

* `syllables.jl` contains a simple implementation of the syllable detection from J. Xu, W. Liao and T. Inoue, "Speech Speed Awareness System for a Non-native Speaker," 2016 International Conference on Collaboration Technologies and Systems (CTS), Orlando, FL, USA, 2016, pp. 43-50, doi: 10.1109/CTS.2016.0027 with CSV outputting. This is WIP and currently doesn't provide the desired results.
* `syllable_detection.py` contains a Python port of the above.
* `noisegatesegments.jl` use a noise gate to detect segments
