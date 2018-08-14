import os
import cv2
from keras.callbacks import Callback
from random import choice
from keras.models import load_model

#Util is for google colab.

LABELS = {0:"Rulo",1:"Normal"}

def beautify_output(label):
	return LABELS[label]



def predict(modelPath,labelFoldersPath):
	
	
	#os.chdir
	#Loads pre-saved model 
	modelBasename = os.path.basename(modelPath)
	modelAbsPath = os.path.join(os.getcwd(), modelBasename)
	model = load_model(modelAbsPath)
	
	while True:
		imgPath = utils.getRandomImage(labelFoldersPath)
		img = cv2.imread(imgPath)
		prediction = model.predict(img)
		cv2.imshow(str(prediction),img)
		# if Property 1 is equal to -1 
		# then window was closed by user.
		while cv2.getWindowProperty(str(prediction),1) != -1:
			

			keyPressed = cv2.waitKey(0)
			if keyPressed  == ord(32):
				break
			elif keyPressed == ord(27):
				exit()
			else:
				continue

	


def getRandomImage(labelFoldersPath):
	labelsDir = os.listdir(".")
	chosenLabel = choice(labelsDir)
	labeledImages = os.listdir(os.path.join(".",chosenLabel))
	chosenImage = choice(labeledImages)
	return os.path.join(chosenLabel,chosenImage)


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
        p1 = subprocess.Popen(['git', 'add', 'checkpoint'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='.')
        p1.wait()
        p2 = subprocess.Popen(['git', 'commit', '-m', "checkpoint updates"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              cwd='.')
        p2.wait()
        p3 = subprocess.Popen(['git', 'push','origin'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd='.')
        p3.wait()
        #print(p3.communicate(), 'i pushed git here')
    except:
        print('i couldn\'t push to git')
        