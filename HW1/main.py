from PyQt5 import QtCore, QtGui, QtWidgets
from HW1 import Ui_MainWindow
import sys
import glob
import numpy as np
import cv2
import hw1_1_FindCorners
import hw1_2
import hw1_5_show
import hw1_5_train
import hw1_5_test


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.HW1_1)
        self.ui.comboBox.addItems(["1", "2", "3", "4", "5"])
        self.ui.pushButton_2.clicked.connect(self.HW1_2)
        self.ui.pushButton_4.clicked.connect(self.HW1_3)
        self.ui.pushButton_3.clicked.connect(self.HW1_4)
        self.ui.pushButton_6.clicked.connect(self.HW2)
        self.ui.pushButton_15.clicked.connect(self.HW5_1)
        self.ui.pushButton_11.clicked.connect(self.HW5_2)
        self.ui.pushButton_12.clicked.connect(self.HW5_3)
        self.ui.pushButton_13.clicked.connect(self.HW5_4)
        self.ui.pushButton_14.clicked.connect(self.HW5_5)
        self.ui.pushButton_10.clicked.connect(self.close)

    def HW1_1(self):
        images = glob.glob("./images/CameraCalibration/*.bmp")
        gray, objpoints, imgpoints = hw1_1_FindCorners.findChessboardCorners(
            images, True
        )
        mtx, dist, rvecs, tvecs = hw1_1_FindCorners.calibrateCamera(
            gray, objpoints, imgpoints
        )
        self.mtx = mtx
        self.dist = dist

    def HW1_2(self):
        print(self.mtx)

    def HW1_3(self):
        images = glob.glob(
            "./images/CameraCalibration/" + str(self.ui.comboBox.currentText()) + ".bmp"
        )
        gray, objpoints, imgpoints = hw1_1_FindCorners.findChessboardCorners(
            images, False
        )
        mtx, dist, rvecs, tvecs = hw1_1_FindCorners.calibrateCamera(
            gray, objpoints, imgpoints
        )
        self.mtx = mtx
        self.dist = dist
        dst, jacobian = cv2.Rodrigues(rvecs)
        print(np.hstack((dst, tvecs[0])))

    def HW1_4(self):
        print(self.dist)

    def HW2(self):
        images = glob.glob("./images/CameraCalibration/1.bmp")
        gray, objpoints, imgpoints = hw1_1_FindCorners.findChessboardCorners(
            images, False
        )
        mtx, dist, rvecs, tvecs = hw1_1_FindCorners.calibrateCamera(
            gray, objpoints, imgpoints
        )
        self.mtx = mtx
        self.dist = dist
        hw1_2.ArgumentedReality(self.mtx, self.dist)

    def HW5_1(self):
        hw1_5_show.show()

    def HW5_2(self):

        print("hyperparameters:")
        print("batch size: 64")
        print("learning rate: 0.001")
        print("optimizer: Adam")

    def HW5_3(self):
        hw1_5_train.train(1)

    def HW5_4(self):
        img = cv2.imread("5_4.png")
        cv2.namedWindow("Image", 0)
        cv2.resizeWindow("Image", 1080, 640)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def HW5_5(self):
        num = int(self.ui.lineEdit_5.text())
        hw1_5_test.test(num)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
