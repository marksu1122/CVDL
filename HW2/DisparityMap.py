import cv2
from matplotlib import pyplot as plt
def stereo():
    imgL = cv2.imread('./imL.png',0)
    imgR = cv2.imread('./imR.png',0)
    #cv2.imshow('imgL', imgL)
    #cv2.imshow('imgR', imgR)
    #numDisparities(int)：即最大視差值與最小視差值之差,窗口大小必須是16的整數倍
    #blockSize：匹配的塊大小。它必須是> = 1的奇數。通常應該在3--11的範圍內。大於11也可以，但必須為奇數。
    stereo = cv2.StereoBM_create(numDisparities=64 , blockSize=9)
    disparity = stereo.compute(imgL,imgR)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.namedWindow("disparity",0)
    cv2.resizeWindow("disparity", 1080, 640)
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()