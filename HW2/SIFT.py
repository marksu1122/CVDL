import cv2
import numpy as np

def keyPoints():
    img1 = cv2.imread('./Aerial1.jpg')
    img2 = cv2.imread('./Aerial2.jpg')
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None) 

    kp1 = sorted(kp1, key = lambda x:x.size, reverse=True)
    kp2 = sorted(kp2, key = lambda x:x.size, reverse=True)

    img1=cv2.drawKeypoints(gray1,kp1[:7], color=(0,0,255), outImage = None)
    img2=cv2.drawKeypoints(gray2,kp2[:7], color=(0,0,255), outImage = None)

    cv2.imshow('Aerial.jpg',img1)
    cv2.imwrite('./Aerial1output.jpg', img1)
    cv2.waitKey(1500)
    cv2.imshow('Aerial.jpg',img2)
    cv2.imwrite('./Aerial2output.jpg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def matchKeypoints():
    img1 = cv2.imread('./Aerial1output.jpg',0) # queryImage
    img2 = cv2.imread('./Aerial2output.jpg',0) # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # des1 = [x for _, x in sorted(zip(kp1,des1), key=lambda pair: pair[0].size)]
    # des2 = [x for _, x in sorted(zip(kp2,des2), key=lambda pair: pair[0].size)]
    # des1 = np.array(des1)
    # des2 = np.array(des2)
    kp1_new = sorted(kp1, key = lambda x:x.size, reverse=True)
    kp2_new = sorted(kp2, key = lambda y:y.size, reverse=True)

    for index in range(7):
        kp1[index] = kp1.index(kp1_new[index])
        kp2[index] = kp2.index(kp2_new[index])

    for index in range(7):
        des1[index] = des1[kp1[index]]
        des2[index] = des2[kp2[index]]
    
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1[:7],des2[:7])
    del matches[3]
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1,kp1_new,img2,kp2_new,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matchKeypoints",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()