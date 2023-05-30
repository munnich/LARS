from typing import Callable, Union
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import QApplication, QMessageBox, QDoubleSpinBox, QSpinBox, QFileDialog, QLineEdit, QMainWindow, QLabel, QGridLayout, QStatusBar, QStyle, QWidget, QToolBar, QProgressBar
from pathlib import Path
import warnings
from numpy._typing import NDArray
import soundfile as sf
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import pandas as pd
from sklearn.neighbors import KDTree
# from syllable_detection import syllable_detection

class AudioParamDoubleSpinBox(QDoubleSpinBox):
    """
    Spin box for audio parameters.
    """
    def __init__(self, parameter: float,
                 update_parameter: Callable,
                 prefix: str,
                 suffix: str,
                 step_size: float = 0.01,
                 param_maximum: Union[float, int] = 1) -> None:
        super().__init__()
        self.setPrefix(prefix)
        self.setSuffix(suffix)
        self.setMinimum(0)
        self.setMaximum(param_maximum)
        self.setValue(parameter)
        self.setSingleStep(step_size)
        self.valueChanged.connect(update_parameter)
        return


class AudioParamSpinBox(QSpinBox):
    """
    Spin box for audio parameters.
    """
    def __init__(self, parameter: int,
                 update_parameter: Callable,
                 prefix: str,
                 suffix: str,
                 step_size: int = 1000,
                 param_maximum: int = int(1e6)) -> None:
        super().__init__()
        self.setPrefix(prefix)
        self.setSuffix(suffix)
        self.setMinimum(0)
        self.setMaximum(param_maximum)
        self.setValue(parameter)
        self.setSingleStep(step_size)
        self.valueChanged.connect(update_parameter)
        return


class Learner():
    """
    Simple k-d tree-based prediction of value corresponding to a vector.
    """
    def __init__(self,
                 vectors: NDArray[np.float64] = np.array([[]], dtype=np.float64),
                 values: NDArray[np.string_] = np.array([], dtype=np.string_),
                 gate: float = 0.01,
                 min_length: int = 2000) -> None:
        """
        :param vectors: Any already determined vectors, usually frequency vectors.
        :param values: Any values corresponding to aforementioned vectors.
        :param gate: Noise gate used for filtering new vectors.
        :param min_length: Minimum length of a non-noise segment.
        """
        self.vectors: NDArray[np.float64] = vectors
        self.values: NDArray[np.string_] = values
        self.gate: float = gate
        self.min_length: int = min_length
        self._regenerate()
        self.tree: Union[None, KDTree] = None
        return

    def _regenerate(self) -> None:
        """
        Regenerate k-d tree.
        """
        if np.size(self.vectors, 1) < 2:
            self.tree: Union[None, KDTree] = None
        else:
            self.tree: Union[None, KDTree] = KDTree(self.vectors)
        return

    def add_vector(self, waveform: NDArray[np.float64], fs: int, value: str) -> None:
        """
        Add new vector to Learner tree.

        :param waveform: Vector's waveform. Spectogram is computed after noise gate filtering.
        :param fs: Sampling frequency.
        :param value: Value corresponding to vector.
        """
        filtered: NDArray[np.float64] = np.array([], dtype=np.float64)
        is_noise: bool = False
        since_noise: int = 0
        # I don't know if this logic works yet
        for i in range(len(waveform)):
            w: float = np.abs(waveform[i])
            if w > self.gate:
                filtered: NDArray[np.float64] = np.append(filtered, w)
                is_noise: bool = False
                since_noise: int = 0
            else:
                if not is_noise:
                    if since_noise > self.min_length:
                        is_noise: bool = True
                        np.append(filtered, waveform[i - since_noise])
                        since_noise: int = 0
                    else:
                        since_noise += 1
        if len(filtered) > fs:
            warnings.warn("Filtered audio vector is greater than sampling frequency and will be trimmed.")
        elif len(filtered) == 0:
            return
        filtered: NDArray[np.float64] = waveform
        freq: NDArray[np.float64] = np.array([np.abs(np.fft.fft(filtered, fs))])
        if len(self.vectors[0]) > 0:
            self.vectors: NDArray[np.float64] = np.concatenate((self.vectors, freq))
        else:
            self.vectors: NDArray[np.float64] = freq
        self.values: NDArray[np.string_] = np.append(self.values, value)
        self._regenerate()
        return

    def predict(self, waveform: NDArray[np.float64]) -> tuple[float, str]:
        """
        Predict a value for given waveform. This is a tree query wrapper.

        The distance returned here can be interpreted as confidence.

        :param waveform: Waveform to match a value for.
        :returns: Tuple of distance and matched value.
        """
        if self.tree is not None:
            dist, i = self.tree.query(waveform.reshape((1, -1)), k=1)
            return dist, self.values[int(i)]
        return np.inf, " "

    def remove_previous(self) -> None:
        """
        Remove the last entry's vector and value from tree.
        """
        self.vectors: NDArray[np.float64] = np.delete(self.vectors, -1)
        self.values: NDArray[np.string_] = np.delete(self.values, -1)
        self._regenerate()
        return


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.title: str = "Label Audio Recording Segments"
        self.abbreviation: str = "LARS"

        # audio parameter defaults
        self.fname: str = ""
        self.fs: int = 44100
        self.audio_full: NDArray[np.float64] = np.array([])
        self.audio: NDArray[np.float64]= np.array([])
        self.frame_length: int = round(self.fs / 2)
        self.overlap: int = 0
        self.position = 0
        self.frames = None
        self.frame_index = 0
        self.fft_display_func = np.abs

        self.learner = None

        # all the labels entered so far
        self.data = pd.DataFrame(columns=["Start", "End", "Labels"])

        self.setWindowTitle(self.abbreviation)

        # self.layout
        self.layout = QGridLayout()

        # current file name
        self.fnameLabel = QLabel("No file selected")

        self.layout.addWidget(self.fnameLabel, 0, 0)

        # parameter spin boxes
        self.frame_length_box = AudioParamSpinBox(self.frame_length, self.update_frame_length, "Frame length: ", " Samples")
        self.layout.addWidget(self.frame_length_box, 1, 0)

        self.overlap_box = AudioParamSpinBox(self.overlap, self.update_overlap, "Overlap: ", " Samples")
        self.layout.addWidget(self.overlap_box, 1, 1)

        # progress bar
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar, 2, 0, 1, 2)

        # plotting objects
        self.graph_widget = pg.PlotWidget()
        self.layout.addWidget(self.graph_widget, 3, 0)

        self.fft_widget = pg.PlotWidget()
        self.layout.addWidget(self.fft_widget, 3, 1)

        # entry
        self.number_of_visible_labels = 70
        self.previous_symbol = ""
        self.previous_text = ""
        self.previous_box = QLabel(self.previous_text)
        self.previous_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout.addWidget(self.previous_box, 4, 0)

        self.entry_box = QLineEdit()
        self.entry_box.setPlaceholderText("Current symbol(s)")
        self.entry_box.returnPressed.connect(self.set_entry)
        self.layout.addWidget(self.entry_box, 4, 1)

        # menu
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        settings_menu = menu.addMenu("Settings")
        help_menu = menu.addMenu("Help")

        # file picking
        file_picker = QAction(self.get_icon("SP_DirOpenIcon"), "Open File", self)
        file_picker.setStatusTip("Choose audio file to process")
        file_picker.triggered.connect(self.on_file_tool_click)
        file_picker.setShortcut(QKeySequence("Ctrl+o"))
        file_menu.addAction(file_picker)

        # save
        save_action = QAction(self.get_icon("SP_DriveFDIcon"), "Save File", self)
        save_action.setStatusTip("Save segment labels to file")
        save_action.triggered.connect(self.save_file)
        save_action.setShortcut(QKeySequence("Ctrl+s"))
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # loading a frames file
        load_frames_action = QAction(self.get_icon("SP_FileDialogDetailedView"), "Load frames",self)
        load_frames_action.setStatusTip("Load frames file")
        load_frames_action.triggered.connect(self.load_frames)
        file_menu.addAction(load_frames_action)

        file_menu.addSeparator()

        # quit
        quit_action = QAction(self.get_icon("SP_TitleBarCloseButton"), "Quit", self)
        quit_action.triggered.connect(QCoreApplication.instance().quit)
        quit_action.triggered.connect(self.close)
        quit_action.setShortcut(QKeySequence("Ctrl+q"))
        file_menu.addAction(quit_action)

        # fft display
        fft_menu = settings_menu.addMenu("FFT Plot")
        self.fft_abs_action = QAction("Magnitude", self)
        self.fft_abs_action.triggered.connect(self.set_fft_abs)
        self.fft_abs_action.setCheckable(True)
        self.fft_abs_action.setChecked(True)
        fft_menu.addAction(self.fft_abs_action)
        self.fft_angle_action = QAction("Phase", self)
        self.fft_angle_action.triggered.connect(self.set_fft_angle)
        self.fft_angle_action.setCheckable(True)
        fft_menu.addAction(self.fft_angle_action)

        # about
        about_action = QAction(f"About {self.abbreviation}", self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        # # how to
        # usageAction = QAction("Usage", self)
        # usageAction.triggered.connect(self.usage)
        # help_menu.addAction(usageAction)

        # toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # file picking
        toolbar.addAction(file_picker)

        # save
        toolbar.addAction(save_action)

        # loading a frames file
        toolbar.addAction(load_frames_action)

        toolbar.addSeparator()

        # play sound
        play_button = QAction(self.get_icon("SP_MediaPlay"), "Play frame", self)
        play_button.setStatusTip("Play audio frame")
        play_button.triggered.connect(self.play_sound)
        play_button.setShortcut(QKeySequence("Ctrl+p"))
        toolbar.addAction(play_button)

        # step backward
        bwd_button = QAction(self.get_icon("SP_MediaSkipBackward"), "Step one frame back", self)
        bwd_button.setStatusTip("Step one frame backwards")
        bwd_button.triggered.connect(self.step_backward)
        bwd_button.setShortcut(QKeySequence("Backspace"))
        toolbar.addAction(bwd_button)

        # learning
        learn_button = QAction(self.get_icon("SP_DriveNetIcon"), "Enable learning", self)
        learn_button.setStatusTip("Enable frequency learning")
        learn_button.setCheckable(True)
        learn_button.triggered.connect(self.toggle_learning)
        toolbar.addAction(learn_button)

        self.learning_gate = 0
        self.learning_gate_box = AudioParamDoubleSpinBox(self.learning_gate, self.update_learning_gate, "Noise gate: ", "", 0.01, 1)
        # self.layout.addWidget(self.learning_gate_box, 5, 0)

        self.distance = 5000
        self.distance_box = AudioParamSpinBox(self.distance, self.update_distance, "Maximum distance: ", "", 50, 10000)
        # self.layout.addWidget(self.distance_box, 5, 1)

        # # automatic estimation via syllable detection (doesn't really work yet)
        # estimationButton = QAction(self.get_icon("SP_BrowserReload"), "Estimate frames", self)
        # estimationButton.setStatusTip("Estimate frames")
        # estimationButton.triggered.connect(self.frames_estimation)
        # toolbar.addAction(estimationButton)

        self.setStatusBar(QStatusBar(self))

        # to container and set as main widget
        container = QWidget()
        container.setLayout(self.layout)

        self.setCentralWidget(container)

        return

    def soft_reset(self) -> None:
        """
        Reset everything other than variables a user would've had to change from the original.
        """
        self.position = 0
        self.previous_symbol = ""
        self.previous_text = ""
        self.graph_widget.clear()
        self.fft_widget.clear()
        self.previous_box.clear()
        self.entry_box.clear()
        self.audio_step()
        return

    def reset(self) -> None:
        """
        Reset all variables.
        """
        self.fname = ""
        self.fs = 44100
        self.audio_full = np.array([])
        self.audio= np.array([])
        self.frame_length = round(self.fs / 2)
        self.overlap = 0
        self.position = 0
        self.previous_symbol = ""
        self.previous_text = ""
        self.graph_widget.clear()
        self.fft_widget.clear()
        self.previous_box.clear()
        return

    def get_icon(self, name) -> QIcon:
        """
        Get Qt icons by name.
        See https://doc.qt.io/qt-6/qstyle.html#StandardPixmap-enum for names.
        """
        return self.style().standardIcon(getattr(QStyle.StandardPixmap, name))

    def on_file_tool_click(self) -> None:
        """
        Open a new audio file.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "",
            "Audio (*.wav *.flac *.opus *.m4a *.ogg *.mp3 *.mka);;Any (*)"
        )
        if filename:
            self.reset()
            path = Path(filename)
            self.fname = path.name
            self.audio_full, self.fs = sf.read(str(path))
            if np.size(self.audio_full, 1) == 2:
                self.audio_full = (self.audio_full[:, 0] + self.audio_full[:, 1]) / 2
            self.fnameLabel.setText(f"Current file: {self.fname} ({self.fs} Hz)")
            self.audio_step()
        return

    def audio_step(self) -> None:
        """
        Step forward one frame length in audio.
        """
        start = max(self.position - self.overlap, 0)
        end = min(self.position + self.frame_length + self.overlap, len(self.audio_full) - 1)
        self.audio = self.audio_full[start:end]
        self.play_sound()
        self.update_plots()
        self.progress_bar.setValue(int((self.position) / (len(self.audio_full) - self.frame_length) * 100))
        if self.learner is not None:
            dist, pred = self.learner.predict(np.abs(np.fft.fft(self.audio, n=self.fs)))
            if dist < self.distance:
                self.entry_box.setText(pred)
        return

    def play_sound(self) -> None:
        """
        Play current frame.
        """
        sd.play(self.audio, self.fs)
        return

    def update_plots(self) -> None:
        """
        Update the plot widgets with current frame.
        """
        self.graph_widget.clear()
        self.graph_widget.plot(np.linspace(max(self.position - self.overlap, 0), min(len(self.audio_full) - 1, self.position + self.frame_length + self.overlap), num=len(self.audio)), self.audio.flatten())
        if self.overlap:
            if self.position > 0:
                self.graph_widget.addItem(pg.InfiniteLine(pos=self.position, label="Previous Frame", labelOpts={"position": 0.1}))
            if self.position + self.frame_length < len(self.audio_full) - 1:
                self.graph_widget.addItem(pg.InfiniteLine(pos=self.position + self.frame_length, label="Next Frame", labelOpts={"position": 0.1}))

        self.fft_widget.clear()
        fftsegment = self.audio_full[self.position:min(len(self.audio_full), self.position + self.frame_length)]
        if len(fftsegment) and max(fftsegment) > 0:
            fft = np.fft.fft(fftsegment, n=self.fs)
            self.fft_widget.plot(self.fft_display_func(fft).flatten()[:round(self.fs / 2)])
        return

    def step_forward(self) -> None:
        """
        Step one frame forward.
        """
        if self.frames is None:
            self.position += self.frame_length
            self.position = min(self.position, len(self.audio_full) - 1)
        else:
            self.frame_index += 1
            if len(self.frames) > self.frame_index:
                self.position = self.frames[self.frame_index][0]
                self.frame_length = self.frames[self.frame_index][1] - self.position
            else:
                self.on_done()
        self.audio_step()
        return

    def step_backward(self) -> None:
        """
        Step one frame backwards, removing the previous text.
        """
        if self.frames is None:
            self.position -= self.frame_length
            self.position = max(self.position, 0)
        else:
            self.frame_index = max(0, self.frame_index - 1)
            self.position = self.frames[self.frame_index][0]
            self.frame_length = self.frames[self.frame_index][1] - self.position
        self.audio_step()
        # temporarily update previous_symbol with last data entry label
        # this enables multiple steps back
        self.previous_symbol = self.data.iloc[-1]["Labels"]
        self.previous_text = self.previous_text[:-len(self.previous_symbol)]
        self.previous_symbol = ""
        self.update_previous()
        self.data = self.data[:-1]
        return

    def update_previous(self) -> None:
        """
        Add new label to displayed text and trim if necessary.
        """
        self.previous_text += self.previous_symbol + "|"
        if len(self.previous_text) > self.number_of_visible_labels:
            self.previous_text = self.previous_text[-self.number_of_visible_labels:]
        self.previous_box.setText(self.previous_text)
        return

    def set_entry(self) -> None:
        """
        Set label for current frame. If current frame is the last frame, a save prompt is opened.
        """
        self.previous_symbol = self.entry_box.text()
        if self.previous_symbol == "":
            self.previous_symbol = " "
        # add to data
        new_row = pd.DataFrame([{"Start": self.position, "End": min(self.position + self.frame_length, len(self.audio_full)), "Labels": self.previous_symbol}])
        self.data = pd.concat([self.data, new_row])
        # add to learner if enabled
        if self.learner is not None:
            self.learner.add_vector(self.audio, self.fs, self.previous_symbol)
        self.update_previous()
        self.entry_box.clear()
        self.step_forward()
        # in this case we're done
        if self.frames is None and self.position == len(self.audio_full) - 1:
            self.on_done()
        return

    def on_done(self) -> None:
        """
        Open save prompt if audio file is fully labeled.
        """
        shouldSave = QMessageBox(self)
        shouldSave.setWindowTitle("Done!")
        shouldSave.setText("You have labeled all segments. Would you like to save your labels to a CSV?")
        shouldSave.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        shouldSave.setIcon(QMessageBox.Icon.Question)
        if shouldSave.exec() == QMessageBox.StandardButton.Yes:
            self.saveCSV()
        return

    def update_frame_length(self) -> None:
        """
        Frame length changed by user.
        """
        self.frame_length = self.frame_length_box.value()
        # I DON'T KNOW IF THIS WILL WORK FINE WILL NEED TO TEST MORE
        self.audio_step()
        # self.soft_reset()
        return

    def update_overlap(self) -> None:
        """
        Overlap changed by user.
        """
        self.overlap = self.overlap_box.value()
        self.audio_step()
        return

    # def frames_estimation(self):
    #     print(syllable_detection(self.audio_full, self.fs))
    #     return

    def about(self) -> None:
        """
        About box.
        Mostly a placeholder.
        """
        QMessageBox.about(self, "About " + self.title, f"{self.title} ({self.abbreviation}) is GNU GPLv3-licensed and was written in Python utilizing the following nonstandard libraries: NumPy, Pandas, PyQt6, pyqtgraph, scipy, sounddevice, soundfile.\n\nAuthors:\nPatrick Munnich.")
        return

    def save_file(self) -> None:
        """
        Save labels to CSV or WAV file.
        """
        # need something here
        # if not self.fname:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            str(Path(self.fname).with_suffix(".csv")),
            "Comma-separated values (*.csv *.wav)"
        )
        if filename:
            if filename.endswith(".csv"):
                self.data.to_csv(Path(filename), index=False)
            elif filename.endswith(".wav"):
                import httpimport
                with httpimport.remote_repo("https://gist.github.com/josephernest/3f22c5ed5dabf1815f16efa8fa53d476/raw/ccea34c6b836fd85c9c0c82d34ffbd02313b30b0"):
                    import wavfile
                markers = [{"position": self.data.iloc[i]["Start"], "label": self.data.iloc[i]["Labels"]} for i in range(self.data.shape[0])]
                wavfile.write(Path(filename), self.fs, self.audio_full, markers=markers)
            else:
                filename += ".csv"
                self.data.to_csv(Path(filename), index=False)
        return

    def load_frames(self) -> None:
        """
        Load a CSV file containing frames for audio file.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Frames File",
            "",
            "Comma-separated values (*.csv)"
        )
        if filename:
            # convert to array of arrays
            self.frames = pd.read_csv(str(Path(filename))).to_numpy(dtype=int)
            self.position = self.frames[0][0]
            self.frame_length = self.frames[0][1] - self.position
            self.audio_step()
        return

    def toggle_learning(self) -> None:
        if self.learner is None:
            self.learner = Learner()
            self.layout.addWidget(self.learning_gate_box, 5, 0)
            self.layout.addWidget(self.distance_box, 5, 1)
        else:
            self.learner = None
            # this is hacky - should be saving a second layout
            self.learning_gate_box.setParent(None)
            self.distance_box.setParent(None)
        return

    def update_learning_gate(self) -> None:
        self.learner.gate = self.learning_gate_box.value()
        return

    def update_distance(self) -> None:
        self.distance = self.distance_box.value()
        return

    def set_fft_abs(self) -> None:
        self.fft_display_func = np.abs
        self.fft_angle_action.setChecked(False)
        self.update_plots()
        return

    def set_fft_angle(self) -> None:
        self.fft_display_func = np.angle
        self.fft_abs_action.setChecked(False)
        self.update_plots()
        return

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
