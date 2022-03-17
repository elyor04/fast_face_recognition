from cv2 import (
    INTER_LINEAR,
    VideoCapture,
    waitKey,
    resize,
    INTER_AREA,
    circle,
    line,
    fillPoly,
    putText,
    FONT_HERSHEY_COMPLEX_SMALL,
    imshow,
    destroyAllWindows,
    drawContours,
)
from face_recognition import face_landmarks
from numpy import array, ndarray
from time import time


def getFixed(lst: list, k=1.0) -> list:
    r_lst = list()
    for val in lst:
        r_lst.append([int(vl * k) for vl in val])
    return r_lst


def getDouble(lst: list, last=True) -> list:
    r_lst, mx = list(), len(lst) - 1
    for i in range(mx + 1):
        if i != mx:
            r_lst.append((lst[i], lst[i + 1]))
        else:
            if last:
                r_lst.append((lst[i], lst[0]))
    return r_lst


def fillPoly_(img, points, color=255) -> ndarray:
    ply = array(points, "int32")
    return fillPoly(img, [ply], color)


def drawContours_(img, points, color=255, thickness=1) -> ndarray:
    cnt = array([[pt] for pt in points], "int32")
    return drawContours(img, [cnt], -1, color, thickness)


def drawLines_(img, points, color=255, thickness=1) -> ndarray:
    points = getDouble(points, False)
    for i, j in points:
        line(img, i, j, color, thickness)
    return img


vdo = VideoCapture(0)
prevTime = 0
waitKey(1000)

while True:
    bgr_img = resize(vdo.read()[1], (0, 0), fx=1.2, fy=1.2, interpolation=INTER_LINEAR)
    rgb_img = resize(bgr_img, (0, 0), fx=0.3, fy=0.3, interpolation=INTER_AREA)[
        :, :, ::-1
    ]
    d_cords = face_landmarks(rgb_img)

    for cords in d_cords:
        for ky in cords.keys():
            crds = getFixed(cords[ky], 3.33)
            if ky in ["left_eye", "right_eye"]:
                fillPoly_(bgr_img, crds, (0, 0, 250))
            elif ky in ["left_eyebrow", "right_eyebrow"]:
                fillPoly_(bgr_img, crds, (5, 5, 5))
            if ky == "chin":
                maska = crds
            else:
                drawContours_(bgr_img, crds, (0, 250, 0), 2)
                for i in crds:
                    circle(bgr_img, i, 1, (250, 0, 0), 2)
        fillPoly_(bgr_img, maska[1:-1], (5, 5, 5))
        drawLines_(bgr_img, maska, (0, 250, 0), 2)
        for i in maska:
            circle(bgr_img, i, 1, (250, 0, 0), 2)

    currTime = time()
    fps = int(1 / (currTime - prevTime))
    prevTime = currTime
    putText(
        bgr_img, f"FPS: {fps}", (5, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, (250, 0, 0), 1
    )
    imshow("BGR video", bgr_img)

    if waitKey(1) & 0xFF == ord("p"):
        n = 0
    else:
        n = 3
    if waitKey(n) & 0xFF == ord("q"):
        break
vdo.release()
destroyAllWindows()
