import numpy as np
import cv2
import glob


def draw(img, corners, imgpts):
    top = tuple(imgpts[0].ravel())
    corner1 = tuple(imgpts[1].ravel())
    corner2 = tuple(imgpts[2].ravel())
    corner3 = tuple(imgpts[3].ravel())
    corner4 = tuple(imgpts[4].ravel())
    # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
    img = cv2.line(img, top, corner1, (0, 0, 255), 3)
    img = cv2.line(img, top, corner2, (0, 0, 255), 3)
    img = cv2.line(img, top, corner3, (0, 0, 255), 3)
    img = cv2.line(img, top, corner4, (0, 0, 255), 3)
    img = cv2.line(img, corner1, corner2, (0, 0, 255), 3)
    img = cv2.line(img, corner2, corner3, (0, 0, 255), 3)
    img = cv2.line(img, corner3, corner4, (0, 0, 255), 3)
    img = cv2.line(img, corner4, corner1, (0, 0, 255), 3)

    return img


def ArgumentedReality(mtx, dist):
    images = glob.glob("./images/CameraCalibration/[1-5].bmp")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[0, 0, -2], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img, corners2, imgpts)
            cv2.namedWindow("img", 0)
            cv2.resizeWindow("img", 1080, 640)
            cv2.imshow("img", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

