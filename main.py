"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10.py
"""
from __future__ import print_function
from argparse import ArgumentParser as argParser
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
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

    sysArgs = argv[1::]
    if sysArgs:
        args = sysArgs

    args = parseArgs(args)
    data_augmentation = args.data_augmentation

    ruloPath = "rulo/valid/"
    normalPath = "normal/valid/"
    paths = [ruloPath,normalPath]
    labels = [1,2]
    
    if args.image_size:
        img_rows = int(args.image_size)
        img_cols = int(int(args.image_size)*4/3)
    
    modelCheckpoint = ModelCheckpoint("./checkpoint")
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.00001, patience=20)
    #tensorboard = TensorBoard()

    batch_size = 32
    nb_classes = 2
    nb_epoch = 200


    img_channels = 3

    
    x_train,y_train,x_test,y_test = utils.load_all_datas(paths,labels)
    
    model = resnet.ResnetBuilder.build_resnet_18((3, 1200, 1600), nb_classes)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=[lr_reducer, early_stopper,modelCheckpoint])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
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

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_test, y_test),
                            epochs=nb_epoch, verbose=1, max_queue_size=100,
                            callbacks=[lr_reducer, early_stopper,modelCheckpoint])

    model.evaluate(x_train,y_train)

if __name__ == "__main__":
    main()