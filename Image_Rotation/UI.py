"""
Created By Cody Kesler
Date: 12/08/2018
CS 501R Final Project
"""

from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
from Model import FinalNet
from Rotate import Rotate

rotate = Rotate()

class Main(QMainWindow):

    def __init__(self):
        super(Main, self).__init__()
        self.initUI()
        self.setWindowTitle('Automatic Image Rotation Correction')
        self.show()

    def initUI(self):
        height_ = 250
        width_ = 600
        self.resize(width_, height_)
        self.center()
        self.text = QLabel(self)
        self.text.setText("Please enter a folder path below. "
                          "\nThe program will auto rotate all pictures in the folder to the correct rotation."
                          "\nThis will use your current user folder as root.")
        self.text.move(100, 0)
        self.text.setMinimumSize(600, 100)

        self.edit = QLineEdit(self)
        self.edit.setText("~/Desktop/pics/")
        self.edit.setMaximumSize(300, 20)
        self.edit.setMinimumSize(300, 20)
        self.edit.move(100, 100)

        self.rotate_button = QPushButton('Auto Rotate', self)
        self.rotate_button.setToolTip('Rotates the Images in the given folder path')
        self.rotate_button.move(100, 150)
        self.rotate_button.clicked.connect(self.rotate)

        self.text2 = QLabel(self)
        self.text2.setText("Progress:")
        self.text2.move(260, 140)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(65, 150, 200, 20)
        self.progress.setMaximum(100)
        self.progress.move(260, 164)

        self.done = DoneWindow(self, "Done Auto Rotating")


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def rotate(self):
        if(rotate.set_path(self.edit.text())):
            self.ext = External()
            self.ext.countChanged.connect(self.onCountChanged)
            self.ext.start()

    def onCountChanged(self, value):
        print(value)
        if (self.progress.value > 99):
            self.done.show()
            self.progress.setValue(0)
        else:
            self.progress.setValue(value)


class External(QThread):
    """
    Runs the parsing of the data
    """
    countChanged = pyqtSignal(int)

    def run(self):
        rotate.run(self.countChanged)



class DoneWindow(QMainWindow):
    def __init__(self, parent=None, title="DONE"):
        super(DoneWindow, self).__init__(parent)
        self.initUI(title)
        self.parent = parent

    def initUI(self, title):
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 200, 100)
        self.setFixedHeight(100)
        self.setFixedWidth(200)
        self.center()

        self.text = QLabel(self)
        self.text.setText("DONE")
        self.text.move(50, 10)

        self.button = QPushButton('OK', self)
        self.button.move(50, 50)
        self.button.clicked.connect(self.done)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def done(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())