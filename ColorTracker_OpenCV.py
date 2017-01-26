# HSV_Color_Extractor
# By: Noureddine AIT SAID
# Contact: noureddine.aitsaid@ieee.org

import cv2
import numpy as np


def nth(x):  # Nothing -> no callback for the Trackbars
    pass

cap = cv2.VideoCapture(0)

# Settings
cv2.namedWindow('Settings', cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Default Green range according to my environment conditions
# min 70    54      0
# max 113   255     255
h, s, v = 70, 54, 0
H, S, V = 113, 255, 255
cv2.namedWindow('Settings', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("H min", 'Settings', h, 179, nth)
cv2.createTrackbar("S min", 'Settings', s, 255, nth)
cv2.createTrackbar("V min", 'Settings', v, 255, nth)

cv2.createTrackbar("H max", 'Settings', H, 179, nth)
cv2.createTrackbar("S max", 'Settings', S, 255, nth)
cv2.createTrackbar("V max", 'Settings', V, 255, nth)

while (1):
    # Take each frame
    _, frame = cap.read()

    # flip it around horizontal axis -> Show it normally -> Convert BGR to HSV to allow color better extraction
    # See the conversion from RGB base to the cylindric HSV base.
    cv2.flip(src=frame, dst=frame, flipCode=1)
    cv2.imshow('Frame', frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # To simplify and to get
    hmin = min(cv2.getTrackbarPos("H min", 'Settings'), cv2.getTrackbarPos("H max", 'Settings'))
    hmax = max(cv2.getTrackbarPos("H min", 'Settings'), cv2.getTrackbarPos("H max", 'Settings'))

    smin = min(cv2.getTrackbarPos("S min", 'Settings'), cv2.getTrackbarPos("S max", 'Settings'))
    smax = max(cv2.getTrackbarPos("S min", 'Settings'), cv2.getTrackbarPos("S max", 'Settings'))

    vmin = min(cv2.getTrackbarPos("V min", 'Settings'), cv2.getTrackbarPos("V max", 'Settings'))
    vmax = max(cv2.getTrackbarPos("V min", 'Settings'), cv2.getTrackbarPos("V max", 'Settings'))

    # define range of blue color in HSV
    lower_bound = np.array([hmin, smin, vmin])
    upper_bound = np.array([hmax, smax, vmax])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphologic ops: erode + dilate -> Eliminate the noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing holes inside the shape
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Bitwise-AND mask and original image -> produces a black image where only the specified color is shown
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Finding Countours
    mask_1 = mask.copy()
    contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_area = 0
        cnt = None
        for i in contours:
            a = cv2.contourArea(i)
            if a > max_area:
                max_area = a
                cnt = i

        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
        cv2.drawContours(res, cnt, -1, (0, 255, 0), 3)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)

    cv2.imshow('Frame', frame)
    cv2.imshow('Result', res)

    k = cv2.waitKey(10)
    # 27 is the ESC button
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
