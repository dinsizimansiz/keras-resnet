"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10.py
"""
from __future__ import print_function
from argparse import ArgumentParser as argParser
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping ,TensorBoard , ModelCheckpoint
from sys import argv
import numpy as np
import resnet
import utils

def parseArgs(args):
	parser = argParser()
	
	parser.add_argument("--data-augmentation",action="store_true")
	parser.add_argument("--image-size",default="1200")
	
	
	
	return parser.parse_args(args)

def main(args = None):
	import os
	sysArgs = argv[1::]
	if sysArgs:
		args = sysArgs

	args = parseArgs(args)
	data_augmentation = args.data_augmentation

	evalPath = os.path.join("images","eval")
	trainPath = os.path.join("images","train")
	
	if args.image_size:
		img_rows = int(args.image_size)
		img_cols = int(int(args.image_size)*4/3)
	
	pushToGitCallback = utils.createPushGitCallback()
	modelCheckpoint = ModelCheckpoint("./checkpoint")
	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	early_stopper = EarlyStopping(min_delta=0.00001, patience=20)
	#tensorboard = TensorBoard()

	batch_size = 32
	nb_classes = 2
	nb_epoch = 200


	img_channels = 3
	train_datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images
	
	eval_datagen = ImageDataGenerator()
	


	model = resnet.ResnetBuilder.build_resnet_50((3, 1200, 1600), 2)
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

	train_generator = train_datagen.flow_from_directory(trainPath,target_size=(1200,1600),class_mode="binary")
	eval_generator = eval_datagen.flow_from_directory(trainPath,target_size=(1200,1600),class_mode="binary")
	model.fit_generator(train_generator,steps_per_epoch=3,epochs=300,validation_data=eval_generator,validation_steps=2,callbacks=[early_stopper,lr_reducer,modelCheckpoint])
	model.save_weights("rasamahram.h5")
if __name__ == "__main__":
	main()