from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QMessageBox, QSpinBox, QFileDialog, QLineEdit, QMainWindow, QLabel, QGridLayout, QStatusBar, QStyle, QWidget, QToolBar, QProgressBar
from pathlib import Path
import soundfile as sf
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import pandas as pd
# from syllable_detection import syllable_detection

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Label Audio Recording Segments"
        self.abbreviation = "LARS"

        # audio parameter defaults
        self.fname = ""
        self.fs = 44100
        self.audio_full = np.array([])
        self.audio= np.array([])
        self.frame_length = round(self.fs / 2)
        self.overlap = 0
        self.position = 0
        self.frames = None
        self.frame_index = 0

        # all the labels entered so far
        self.data = pd.DataFrame(columns=["Start", "End", "Labels"])

        self.setWindowTitle(self.abbreviation)

        # layout
        layout = QGridLayout()

        # current file name
        self.fnameLabel = QLabel("No file selected")

        layout.addWidget(self.fnameLabel, 0, 0)

        # parameter spin boxes
        self.frameLengthBox = QSpinBox()
        self.frameLengthBox.setPrefix("Frame length: ")
        self.frameLengthBox.setSuffix(" Samples")
        self.frameLengthBox.setMinimum(0)
        self.frameLengthBox.setMaximum(50000)
        self.frameLengthBox.setValue(self.frame_length)
        self.frameLengthBox.valueChanged.connect(self.update_frame_length)
        layout.addWidget(self.frameLengthBox, 1, 0)

        self.overlapBox = QSpinBox()
        self.overlapBox.setPrefix("Overlap: ")
        self.overlapBox.setSuffix(" Samples")
        self.overlapBox.setMinimum(0)
        self.overlapBox.setMaximum(50000)
        self.overlapBox.setValue(self.overlap)
        self.overlapBox.valueChanged.connect(self.update_overlap)
        layout.addWidget(self.overlapBox, 1, 1)

        # progress bar
        self.progressBar = QProgressBar(self)
        layout.addWidget(self.progressBar, 2, 0, 1, 2)

        # plotting objects
        self.graphWidget = pg.PlotWidget()
        layout.addWidget(self.graphWidget, 3, 0)

        self.fftWidget = pg.PlotWidget()
        layout.addWidget(self.fftWidget, 3, 1)

        # entry
        self.numberOfVisibleLabels = 70
        self.previousSymbol = ""
        self.previousText = ""
        self.previousBox = QLabel(self.previousText)
        self.previousBox.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.previousBox, 4, 0)

        self.entryBox = QLineEdit()
        self.entryBox.setPlaceholderText("Current symbol(s)")
        self.entryBox.returnPressed.connect(self.set_entry)
        layout.addWidget(self.entryBox, 4, 1)

        # menu
        menu = self.menuBar()
        fileMenu = menu.addMenu("File")
        helpMenu = menu.addMenu("Help")

        # file picking
        filePicker = QAction(self.get_icon("SP_DirOpenIcon"), "Open File", self)
        filePicker.setStatusTip("Choose audio file to process")
        filePicker.triggered.connect(self.onFileToolClick)
        filePicker.setShortcut(QKeySequence("Ctrl+o"))
        fileMenu.addAction(filePicker)

        # save
        saveAction = QAction(self.get_icon("SP_DriveFDIcon"), "Save File", self)
        saveAction.setStatusTip("Save segment labels to CSV")
        saveAction.triggered.connect(self.saveCSV)
        saveAction.setShortcut(QKeySequence("Ctrl+s"))
        fileMenu.addAction(saveAction)

        fileMenu.addSeparator()

        # loading a frames file
        loadFramesAction = QAction(self.get_icon("SP_FileDialogDetailedView"), "Load frames",self)
        loadFramesAction.setStatusTip("Load frames file")
        loadFramesAction.triggered.connect(self.load_frames)
        fileMenu.addAction(loadFramesAction)

        fileMenu.addSeparator()

        # quit
        quitAction = QAction(self.get_icon("SP_TitleBarCloseButton"), "Quit", self)
        quitAction.triggered.connect(QCoreApplication.instance().quit)
        quitAction.triggered.connect(self.close)
        quitAction.setShortcut(QKeySequence("Ctrl+q"))
        fileMenu.addAction(quitAction)

        # about
        aboutAction = QAction(f"About {self.abbreviation}", self)
        aboutAction.triggered.connect(self.about)
        helpMenu.addAction(aboutAction)

        # # how to
        # usageAction = QAction("Usage", self)
        # usageAction.triggered.connect(self.usage)
        # helpMenu.addAction(usageAction)

        # toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # file picking
        toolbar.addAction(filePicker)

        # save
        toolbar.addAction(saveAction)

        # loading a frames file
        toolbar.addAction(loadFramesAction)

        toolbar.addSeparator()

        # play sound
        playButton = QAction(self.get_icon("SP_MediaPlay"), "Play frame", self)
        playButton.setStatusTip("Play audio frame")
        playButton.triggered.connect(self.play_sound)
        playButton.setShortcut(QKeySequence("Ctrl+p"))
        toolbar.addAction(playButton)

        # step backward
        bwdButton = QAction(self.get_icon("SP_MediaSkipBackward"), "Step one frame back", self)
        bwdButton.setStatusTip("Step one frame backwards")
        bwdButton.triggered.connect(self.step_backward)
        bwdButton.setShortcut(QKeySequence("Backspace"))
        toolbar.addAction(bwdButton)

        # # automatic estimation via syllable detection (doesn't really work yet)
        # estimationButton = QAction(self.get_icon("SP_BrowserReload"), "Estimate frames", self)
        # estimationButton.setStatusTip("Estimate frames")
        # estimationButton.triggered.connect(self.frames_estimation)
        # toolbar.addAction(estimationButton)

        self.setStatusBar(QStatusBar(self))

        # to container and set as main widget
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        return

    def soft_reset(self):
        """
        Reset everything other than variables a user would've had to change from the original.
        """
        self.position = 0
        self.previousSymbol = ""
        self.previousText = ""
        self.graphWidget.clear()
        self.fftWidget.clear()
        self.previousBox.clear()
        self.entryBox.clear()
        self.audio_step()
        return

    def reset(self):
        """
        Reset all variables.
        """
        self.fname = ""
        self.fs = 44100
        self.audio_full = np.array([])
        self.audio= np.array([])
        self.frame_length = round(self.fs)
        self.overlap = 0
        self.position = 0
        self.previousSymbol = ""
        self.previousText = ""
        self.graphWidget.clear()
        self.fftWidget.clear()
        self.previousBox.clear()
        return

    def get_icon(self, name):
        """
        Get Qt icons by name.
        See https://doc.qt.io/qt-6/qstyle.html#StandardPixmap-enum for names.
        """
        return self.style().standardIcon(getattr(QStyle.StandardPixmap, name))

    def onFileToolClick(self):
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

    def audio_step(self):
        """
        Step forward one frame length in audio.
        """
        start = max(self.position - self.overlap, 0)
        end = min(self.position + self.frame_length + self.overlap, len(self.audio_full) - 1)
        self.audio = self.audio_full[start:end]
        self.play_sound()
        self.update_plots()
        self.progressBar.setValue(int((self.position) / (len(self.audio_full) - self.frame_length) * 100))
        return

    def play_sound(self):
        """
        Play current frame.
        """
        sd.play(self.audio, self.fs)
        return

    def update_plots(self):
        """
        Update the plot widgets with current frame.
        """
        self.graphWidget.clear()
        self.graphWidget.plot(np.linspace(max(self.position - self.overlap, 0), min(len(self.audio_full) - 1, self.position + self.frame_length + self.overlap), num=len(self.audio)), self.audio.flatten())
        if self.overlap:
            if self.position > 0:
                self.graphWidget.addItem(pg.InfiniteLine(pos=self.position, label="Previous Frame", labelOpts={"position": 0.1}))
            if self.position + self.frame_length < len(self.audio_full) - 1:
                self.graphWidget.addItem(pg.InfiniteLine(pos=self.position + self.frame_length, label="Next Frame", labelOpts={"position": 0.1}))

        self.fftWidget.clear()
        fftsegment = self.audio_full[self.position:min(len(self.audio_full), self.position + self.frame_length)]
        if max(fftsegment) > 0:
            self.fftWidget.plot(np.abs(np.fft.fft(fftsegment, n=self.fs)).flatten()[:round(self.fs / 2)])
        return

    def step_forward(self):
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

    def step_backward(self):
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
        # temporarily update previousSymbol with last data entry label
        # this enables multiple steps back
        self.previousSymbol = self.data.iloc[-1]["Labels"]
        self.previousText = self.previousText[:-len(self.previousSymbol)]
        self.previousSymbol = ""
        self.update_previous()
        self.data = self.data[:-1]
        return

    def update_previous(self):
        """
        Add new label to displayed text and trim if necessary.
        """
        self.previousText += self.previousSymbol + "|"
        if len(self.previousText) > self.numberOfVisibleLabels:
            self.previousText = self.previousText[-self.numberOfVisibleLabels:]
        self.previousBox.setText(self.previousText)
        return

    def set_entry(self):
        """
        Set label for current frame. If current frame is the last frame, a save prompt is opened.
        """
        self.previousSymbol = self.entryBox.text()
        if self.previousSymbol == "":
            self.previousSymbol = " "
        new_row = pd.DataFrame([{"Start": self.position, "End": min(self.position + self.frame_length, len(self.audio_full)), "Labels": self.previousSymbol}])
        self.data = pd.concat([self.data, new_row])
        self.update_previous()
        self.entryBox.clear()
        self.step_forward()
        # in this case we're done
        if self.frames is None and self.position == len(self.audio_full) - 1:
            self.on_done()
        return

    def on_done(self):
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

    def update_frame_length(self):
        """
        Frame length changed by user.
        """
        self.frame_length = self.frameLengthBox.value()
        self.soft_reset()
        return

    def update_overlap(self):
        """
        Overlap changed by user.
        """
        self.overlap = self.overlapBox.value()
        self.audio_step()
        return

    # def frames_estimation(self):
    #     print(syllable_detection(self.audio_full, self.fs))
    #     return

    def about(self):
        """
        About box.
        Mostly a placeholder.
        """
        QMessageBox.about(self, "About " + self.title, f"{self.title} ({self.abbreviation}) is GNU GPLv3-licensed and was written in Python utilizing the following nonstandard libraries: NumPy, Pandas, PyQt6, pyqtgraph, scipy, sounddevice, soundfile.\n\nAuthors:\nPatrick Munnich.")
        return

    def saveCSV(self):
        """
        Save labels to CSV file.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            str(Path(self.fname).with_suffix(".csv")),
            "Comma-separated values (*.csv)"
        )
        if filename:
            if not filename.endswith(".csv"):
                filename += ".csv"
            self.data.to_csv(Path(filename), index=False)
        return

    def load_frames(self):
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

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
