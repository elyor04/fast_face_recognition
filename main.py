from face_recog import FaceRecog, YigibTekshir
from cv2 import (
    VideoCapture,
    resize,
    INTER_AREA,
    rectangle,
    getTextSize,
    putText,
    FONT_HERSHEY_COMPLEX_SMALL,
    imshow,
    waitKey,
    destroyAllWindows,
)
from threading import Timer
from time import time


def getFixed(lst: list, k: float = 1.0) -> list:
    r_lst = list()
    for val in lst:
        r_lst.append([int(vl * k) for vl in val])
    return r_lst


data = [("photos/biden.jpg", "Biden"), ("photos/obama.jpg", "Obama")]
oby = FaceRecog(data)
vdo = VideoCapture(0)
waitKey(1000)

mlmt: list[tuple[tuple, str]]
pp: list[tuple[int, Timer]]
oby2, mlmt = YigibTekshir(), list()
pp, inds = list(), list()
prevTime = 0

while True:
    bgr_img, n = vdo.read()[1], 3
    rgb_img = resize(bgr_img, (0, 0), fx=0.4, fy=0.4, interpolation=INTER_AREA)[
        :, :, ::-1
    ]
    oby.start(rgb_img, mlmt)
    locs = oby.getLocations()
    names = oby.getNames()
    anqlr = oby2.start(names)

    inds.clear()
    for i in range(len(mlmt)):
        if mlmt[i][1] not in names:
            inds.insert(0, i)
        else:
            anq = mlmt[i][1]
            ind = names.index(anq)
            mlmt[i] = (locs[ind], anq)
            for j in pp:
                if j[0] == i:
                    j[1].cancel()
                    pp.remove(j)

    for i in inds:
        for j in pp:
            if j[0] == i:
                break
        else:
            tm = Timer(1.0, mlmt.pop, args=(i,))
            tm.start()
            pp.append((i, tm))

    for anq in anqlr:
        if (anq != "?") and (anq in names):
            for i in mlmt:
                if i[1] == anq:
                    break
            else:
                ind = names.index(anq)
                mlmt.append((locs[ind], anq))

    for (y, x2, y2, x), name in zip(getFixed(locs, 2.5), names):
        rectangle(bgr_img, (x, y), (x2, y2), (200, 0, 0), 2)
        u = getTextSize(name, FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1)
        u = max(u[0][0] + x, x2)
        bgr_img[y2 : y2 + 28, x - 2 : u + 2] = (200, 0, 0)
        putText(
            bgr_img, name, (x, y2 + 18), FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 250, 0), 1
        )

    currTime = time()
    fps = int(1 / (currTime - prevTime))
    prevTime = currTime
    putText(
        bgr_img, f"FPS: {fps}", (5, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, (250, 0, 0), 1
    )
    imshow("yuzni tanish", bgr_img)

    if waitKey(1) & 0xFF == ord("p"):
        n = 0
    if waitKey(n) & 0xFF == ord("q"):
        break
vdo.release()
destroyAllWindows()
