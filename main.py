"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10.py
"""
from __future__ import print_function
import tensorboard
from argparse import ArgumentParser as argParser
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.applications import ResNet50 as Resnet50
from sys import argv
import numpy as np
import utils



def parseArgs(args):
	parser = argParser()	
	parser.add_argument("--data-augmentation", action="store_true")
	parser.add_argument("--image-size", default="1200")
	parser.add_argument("--new-training",action="store_true")
	parser.add_argument("--steps",default="200")
	parser.add_argument("--epochs",default="300")
	parser.add_argument("--lr",default="0.00001")
	parser.add_argument("--gitpush",action="store_true")
	parser.add_argument("--test",action="store_true")
	parser.add_argument("--test-dir",default=None)
	parser.add_argument("--alltrain",action="store_true")
	return parser.parse_args(args)

#
#
#
#
#
# /label1/image1.jpg
# /label2/image2.jpg
# Outputs label1 or label2
#
#
#
#


def main(args=None):
	import os
	sysArgs = argv[1::]
	if sysArgs:
		args = sysArgs

	evalPath = os.path.join("images", "eval")
	trainPath = os.path.join("images", "train")
	checkpointPath = os.path.join("checkpoint","model_ckpt.h5")
	args = parseArgs(args)
	data_augmentation = args.data_augmentation
	learning_rate = float(args.lr)
	number_of_steps = int(args.steps)
	number_of_epochs = int(args.epochs)
	if args.test:
		if not args.test_dir :
			raise Exception("Test directory is not specified.")
		else:
			test_dir = os.path.join(args.test_dir)
			utils.predict(checkpointPath,test_dir)
			

	if args.image_size:
		img_rows = int(args.image_size)
		img_cols = int(int(args.image_size) * 4 / 3)

	pushToGitCallback = utils.createPushGitCallback()
	modelCheckpoint = ModelCheckpoint(os.path.join("checkpoint","model_ckpt.h5"),verbose=1)
	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	early_stopper = EarlyStopping(min_delta=0.00001, patience=20)
	board = TensorBoard(log_dir=os.path.join("tensorlog"),histogram_freq=1,write_graph=True)
	callbacks = [modelCheckpoint]

	if args.gitpush:
		callbacks.append(pushToGitCallback)
	batch_size = 1
	nb_classes = 2
	nb_epoch = 200
	imageSize = (1200, 1600)
	
	
	
	if args.alltrain:
		evalPath = os.path.join("images", "all")
		trainPath = os.path.join("images", "all")
	
		
		
		
	img_channels = 3
	train_datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,#True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	eval_datagen = ImageDataGenerator()
	train_generator = train_datagen.flow_from_directory(trainPath, target_size=imageSize, class_mode="binary", batch_size=batch_size)
	eval_generator = eval_datagen.flow_from_directory(evalPath, target_size=imageSize, class_mode="binary")

	
	optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	if args.new_training:
		model = Resnet50(weights=None,input_shape=(*imageSize,3), classes=2)
		model.compile(loss='binary_crossentropy',
					  optimizer=optimizer,
					  metrics=['acc'])

		model.fit_generator(train_generator, steps_per_epoch=number_of_steps, epochs=number_of_epochs, validation_data=train_generator,
							validation_steps=20, callbacks=callbacks)#early_stopper, lr_reducer, modelCheckpoint
	
		
	else:
		try:

			model = load_model(os.path.join("checkpoint","model_ckpt.h5"))
			model.fit_generator(train_generator, steps_per_epoch=number_of_steps, epochs=number_of_epochs, validation_data=train_generator,
							validation_steps=20, callbacks=callbacks)#early_stopper, lr_reducer, modelCheckpoint
		except:
			model = Resnet50(weights=None,input_shape=(*imageSize,3), classes=2)
			model.compile(loss='binary_crossentropy',
					  optimizer=optimizer,
					  metrics=['acc'])

			model.fit_generator(train_generator, steps_per_epoch=number_of_steps, epochs=number_of_epochs, validation_data=train_generator,
							validation_steps=20, callbacks=callbacks)#early_stopper, lr_reducer, modelCheckpoint
	
		
if __name__ == "__main__":
#	main(["--test","--test-dir","images"])
	main(["--new-training"])
	main()