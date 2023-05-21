import cv2
import numpy as np
import os


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extractor(img_path, output_path):
    per = 25
    roi = [[(900, 600), (1200, 735), 'box', 'signature'], ]

    imgQ = cv2.imread('test_2.jpeg')
    h, w, c = imgQ.shape

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    img = cv2.imread(img_path)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.match(des2, des1)

    """Prints the matches in the console"""
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    srcPoints = np.float32(
        [kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32(
        [kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (imgQ.shape[1], imgQ.shape[0]))

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]),
                      (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        # Cropped Images of the signatures
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        bg_img = cv2.imread('bd_img.jpg')

        hsv = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 0, 0])
        upper = np.array([120, 150, 120])

        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        sign_mask = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)

        height, width, channels = imgCrop.shape

        placeToPutSign = bg_img[0:height, 0:width]
        placeToPutSign_mask = cv2.bitwise_and(
            placeToPutSign, placeToPutSign, mask=mask_inv)
        placeToPutSign_joined = cv2.add(placeToPutSign_mask, sign_mask)

        create_dir('Results/')
        cv2.imwrite('Results/' + "newImg.jpeg", placeToPutSign_mask)
