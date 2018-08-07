import os
import cv2
def partition(lst, n = 2): 
    from random import shuffle
    shuffle(lst)
    division = len(lst) / float(n) 
    return tuple([ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ])
    

def getAllFiles(path=None):
    
    return [os.path.join(os.getcwd(),path,filename) for filename in os.listdir(path)]

def labelize(lst:list,label):
    return [(elem,label) for elem in lst] 


def load_img(path):
    #os.getcwd()
    return cv2.imread("/"+path)

def load_data_from_folder(path):
    nameOfImages = os.listdir(path)
    data = []
    for picName in nameOfImages:
        absPath = os.path.join(os.getcwd(),path,picName)
        imgData = load_img(absPath)
        data.append(imgData)
    return data

def load_data(path,label):
    data = load_data_from_folder(path)
    return labelize(data,label)

#
#   @returns{[(numpy.ndarray,labelType)]}
#
#   Returns list of pairs
#   First element of pair includes image
#   Second element of pair includes label.
#
def merge_data(pathList:list,labelList:list):
    pairs = []
    for path,label in zip(pathList,labelList):
        loaded_data = load_data(path,label)
        pairs = list(pairs + loaded_data)
    return pairs


#
#
#   @returns{(x_train,y_train),(x_test,y_test)}
#
#
def load_all_datas(pathList,labelList):
    merged_data = merge_data(pathList,labelList)
    return partition(merged_data)    
    
    
    
    