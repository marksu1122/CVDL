from PyQt5 import QtCore, QtGui, QtWidgets
from HW2 import Ui_MainWindow
import sys
import glob
import numpy as np
import cv2
import DisparityMap
import NCC
import SIFT



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.HW1)
        self.ui.pushButton_2.clicked.connect(self.HW2)
        self.ui.pushButton_3.clicked.connect(self.HW3_1)
        self.ui.pushButton_4.clicked.connect(self.HW3_2)
        self.ui.pushButton_5.clicked.connect(self.close)

    def HW1(self):
        DisparityMap.stereo()

    def HW2(self):
        NCC.NCC()

    def HW3_1(self):
        SIFT.keyPoints()
        
    def HW3_2(self):
        SIFT.matchKeypoints()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
