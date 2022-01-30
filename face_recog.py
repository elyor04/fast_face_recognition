from face_recognition import (
    load_image_file, face_locations, face_encodings, face_distance)
from cv2 import (
    resize, INTER_AREA, INTER_LINEAR, INTER_CUBIC)
from numpy import argmin, ndarray
from math import sqrt

def sozQiymat(img:ndarray=None, eni:int=None, buyi:int=None, yuza:int=100000) -> tuple[int,int]:
    if (eni is None) or (buyi is None):
        buyi, eni = img.shape[:2]
    k = round(sqrt(yuza / (eni * buyi)), 4)
    x, y = int(eni * k), int(buyi * k)
    return (x, y)

def sozlash(img:ndarray, eni:int=None, buyi:int=None, sifat:bool=False, auto:bool=False, yuza:int=100000) -> ndarray:
    if auto: x, y = sozQiymat(img, yuza=yuza)
    else: x, y = eni, buyi
    buyi, eni = img.shape[:2]

    if x <= eni and y <= buyi:
        return resize(img, (x, y), interpolation=INTER_AREA)
    elif x >= eni and y >= buyi:
        if sifat:
            return resize(img, (x, y), interpolation=INTER_CUBIC)
        else:
            return resize(img, (x, y), interpolation=INTER_LINEAR)
    else:
        return resize(img, (x, y))

def getCenter(cord: tuple) -> tuple[int,int]:
    y, x2, y2, x = cord
    x = int((x + x2) / 2)
    y = int((y + y2) / 2)
    return (x, y)

class FaceRecog(object):
    def __init__(self, file_name:list=None, faceEncod_name:list=None) -> None:
        super(FaceRecog, self).__init__()
        self.__f_encs, self.__f_nmes = list(), list()
        if file_name is not None:
            for f, n in file_name:
                prsn = sozlash(load_image_file(f), auto=True)
                self.__f_encs.append(face_encodings(prsn)[0])
                self.__f_nmes.append(n)
        else:
            for fe, n in faceEncod_name:
                self.__f_encs.append(fe)
                self.__f_nmes.append(n)
        self.__ln_f, self.__centrs = len(self.__f_encs), list()
        self.__locs, self.__names = list(), list()
    
    def __sortLocs(self, locs:list) -> list:
        rtrn, locs = list(), locs.copy()
        pp = list()
        for ind, (x,y) in enumerate(self.__centrs):
            for ind2, loc in enumerate(locs):
                x2, y2 = getCenter(loc)
                if (abs(x-x2) < 20) and (abs(y-y2) < 20):
                    rtrn.append(loc)
                    locs.pop(ind2)
                    self.__centrs[ind] = (x2, y2)
                    break
            else: pp.insert(0, ind)
        for i in pp: self.__centrs.pop(i)
        for loc in locs:
            rtrn.append(loc)
            self.__centrs.append(getCenter(loc))
        return rtrn
    
    def reset(self, file_name:list=None, faceEncod_name:list=None) -> None:
        self.__init__(file_name, faceEncod_name)
    
    def appendFace(self, file_name:tuple=None, faceEncod_name:tuple=None) -> None:
        if file_name is not None:
            f, n = file_name
            prsn = sozlash(load_image_file(f), auto=True)
            self.__f_encs.append(face_encodings(prsn)[0])
            self.__f_nmes.append(n)
        else:
            fe, n = faceEncod_name
            self.__f_encs.append(fe)
            self.__f_nmes.append(n)
        self.__ln_f += 1
    
    def removeFace(self, name:str=None, index:int=None) -> None:
        if (name is not None) and (name in self.__f_nmes):
            index = self.__f_nmes.index(name)
        self.__f_encs.pop(index)
        self.__f_nmes.pop(index)
        self.__ln_f -= 1

    def start(self, rgb_img:ndarray, infos:list=[]) -> None:
        self.__names.clear()
        self.__locs = self.__sortLocs(face_locations(rgb_img))

        if self.__ln_f != 0:
            inf_ln = len(infos)
            if inf_ln != 0:
                lcs, ind_nme, infos = list(), list(), infos.copy()
                for ind, loc in enumerate(self.__locs):
                    x, y = getCenter(loc)
                    for ind2, info in enumerate(infos):
                        (x2, y2), nm = getCenter(info[0]), info[1]
                        if (abs(x-x2) < 20) and (abs(y-y2) < 20):
                            ind_nme.append((ind, nm))
                            infos.pop(ind2)
                            break
                    else: lcs.append(loc)
            else: lcs = self.__locs

            encs = face_encodings(rgb_img, lcs)
            for enc in encs:
                dis = face_distance(self.__f_encs, enc)
                ind = argmin(dis)
                if dis[ind] <= 0.4:
                    self.__names.append(self.__f_nmes[ind])
                else: self.__names.append("notanish")
            if inf_ln != 0:
                for i, j in ind_nme:
                    self.__names.insert(i, j)
        else:
            self.__names = ["notanish" for i in range(len(self.__locs))]

    def getNames(self) -> list:
        return self.__names
    def getLocations(self) -> list:
        return self.__locs

class YigibTekshir(object):
    def __init__(self) -> None:
        super(YigibTekshir, self).__init__()
        self.__ismlar: list[list]
        self.__ismlar = list()
    
    def start(self, texts:list) -> list:
        rtrn = list()
        if len(texts) == 0:
            self.__ismlar.clear()
        for i in range(len(texts)-len(self.__ismlar)):
            self.__ismlar.append([])
        
        for i in range(len(texts)):
            self.__ismlar[i].append(texts[i])
            ism_ln, aniq = len(self.__ismlar[i]), '?'
            if ism_ln >= 4:
                for j in self.__ismlar[i]:
                    foiz = (self.__ismlar[i].count(j) / ism_ln) * 100
                    if foiz > 60:
                        aniq = j
                        self.__ismlar[i].clear()
                        break
            rtrn.append(aniq)
            if len(self.__ismlar[i]) == 8:
                self.__ismlar[i].clear()
        return rtrn
