import os
import cv2
from keras.callbacks import Callback




createPushGitCallback = lambda : PushGitCallback()
  
        
    
class PushGitCallback(Callback):
        
        
    def __init__(self):
        super(PushGitCallback,self).__init__()
    
    
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
        p1 = subprocess.Popen(['git', 'add', 'checkpoint/*'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='/keras-resnet')
        p1.wait()
        p2 = subprocess.Popen(['git', 'commit', '-m', "snapshot-update"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              cwd='/keras-resnet')
        p2.wait()
        p3 = subprocess.Popen(['git', 'push'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='/keras-resnet')
        p3.wait()
        #print(p3.communicate(), 'i pushed git here')
    except:
        print('i couldn\'t push to git')
        