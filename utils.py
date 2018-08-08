import os
import cv2
from keras.callbacks import Callback
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
        pairs = pairs+loaded_data
    return pairs


#
#
#   @returns{(x_train,y_train),(x_test,y_test)}
#
#
def load_all_datas(pathList,labelList):
    merged_data = merge_data(pathList,labelList)
    test,train = partition(merged_data)    
    x_test = list()
    y_test = list()
    x_train = list()
    y_train = list()
    
    for arr,label in test:
        x_test.append(arr)
        y_test.append(label)
    for arr,label in train:
        x_train.append(arr)
        y_train.append(label)
    
    return (x_train,y_train,x_test,y_test)


createPushGitCallback = lambda : PushGitCallback()
  
        
    
class PushGitCallback(Callback):
        
        
    def __init__(self):
        super(PushGitCallback,self)
    
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pushToGitAsync()


def pushToGitAsync():
    from time import time
    startTime = time()
    from threading import Thread
    thread = Thread(target=pushToGit)
    thread.start()
    print('i pushed git here, seconds:', time() - startTime)


def pushToGit():
    try:
        #print('i will push git here')
        import subprocess
        p1 = subprocess.Popen(['git', 'add', 'snapshots/*'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='/content/medikal-ml')
        p1.wait()
        p2 = subprocess.Popen(['git', 'commit', '-m', "snapshot-update"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              cwd='/content/medikal-ml')
        p2.wait()
        p3 = subprocess.Popen(['git', 'push'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='/content/medikal-ml')
        p3.wait()
        #print(p3.communicate(), 'i pushed git here')
    except:
        print('i couldn\'t push to git')
        