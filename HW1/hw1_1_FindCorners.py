import numpy as np
import cv2
import glob


#對每張圖片，識別出角點，記錄世界物體座標和影象座標
def findChessboardCorners(images,show):
    # termination criteria
    #criteria:角點精準化迭代過程的終止條件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    objp = np.zeros((8*11,3), np.float32)
    '''
    設定世界座標下點的座標值，因為用的是棋盤可以直接按網格取；
    假定棋盤正好在x-y平面上，這樣z值直接取0，簡化初始化步驟。
    mgrid把列向量[0:cbraw]複製了cbcol列，把行向量[0:cbcol]複製了cbraw行。
    轉置reshape後，每行都是4×6網格中的某個點的座標。
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        #轉灰度
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # 尋找角點，存入corners，ret是找到角點的flag
        ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            #執行亞畫素級(更準確))角點檢測
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            if show:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
                cv2.namedWindow("findCorners",0);
                cv2.resizeWindow("findCorners", 1080, 640);
                cv2.imshow('findCorners',img)
                cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    return gray,objpoints,imgpoints
# Calibrate Camera    
def calibrateCamera(gray,objpoints,imgpoints):
    """
    傳入所有圖片各自角點的三維、二維座標，相機標定。
    每張圖片都有自己的旋轉和平移矩陣，但是相機內參和畸變係數只有一組。
    mtx，相機內參；dist，畸變係數；revcs，旋轉矩陣；tvecs，平移矩陣。
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    mtx = np.array(mtx)
    dist = np.array(dist)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return mtx,dist,rvecs,tvecs

