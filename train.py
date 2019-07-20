# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:38:01 2019

@author: ECO-EMS
"""
import random
from sklearn.model_selection import train_test_split
import argparse
import os
import glob
import tensorflow as tf
import cv2
import numpy as np
from model import *

def create_y(filename):
    img = cv2.imread(filename.decode(), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = img.shape[:2]
    rand_y_shift = random.randint(-int(h*0.05/2), int(h*0.05/2))
    rand_x_shift = random.randint(-int(w*0.05/2), int(w*0.05/2))

    center = (w / 2 + rand_x_shift, h / 2+rand_y_shift)
    angle_rnd = random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle_rnd, 1.0)
    rotated_random = cv2.warpAffine(gray, M, (h, w))
    gray = cv2.resize(gray, (IMG_HEIGHT, IMG_WIDTH))
    rotated_random = cv2.resize(rotated_random, (IMG_HEIGHT, IMG_WIDTH))
    rotated_random = rotated_random.reshape((IMG_HEIGHT, IMG_WIDTH,1))
    gray = rotated_random.reshape((IMG_HEIGHT, IMG_WIDTH,1))

    outp = np.concatenate((gray, rotated_random), axis=2)
    
    return outp.astype(np.float32), np.float32(float(angle_rnd)/360.0)
    
def parse_arguments():
    ## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Train the ML model.",
                        action="store_true")
    
    parser.add_argument("-ip", "--img_path", help="File path for images. Default is .\\data\\images. Please use \\ defining the path...",
                        default=".\\images\\")  
    
    parser.add_argument("-op", "--output_path", help="Path to save ML models.",
                        default=".\\")   
    
    parser.add_argument("-p", "--predict", help="Predict image angle. The input images should be in RGB format",
                        action="store_true")
    
    parser.add_argument("-ia", "--imageA", help="Image 1",
                        default="")   

    parser.add_argument("-ib", "--imageB", help="Image 2",
                        default="")    
    parser.add_argument("-fm", "--filemeta", help="Checkpoint metadata",
                        default="")     
    args = parser.parse_args()
    
    if (args.train==False and args.predict==False) or  (args.train==True and args.predict==True):
        print("Please select one of the two functionalities!")
        parser.print_help()
        os._exit(0)
        
    if args.predict==True and (args.imageA=="" or args.imageB==""):
        print("Please give both image file paths!")
        parser.print_help()
        os._exit(0)    
        
    if args.predict==True and (os.path.isfile(args.imageA)==False or os.path.isfile(args.imageA)==False):
        print("Please give correct image file paths!")
        parser.print_help()
        os._exit(0)    
        
        
    return args    

if __name__ == "__main__":
    args = parse_arguments()
    
    
    if args.predict:
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(args.filemeta)
            img = cv2.imread(args.imageA, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sess.run(init_op)

            saver.restore(sess, tf.train.latest_checkpoint("./"))
            img = cv2.imread(args.imageA, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_HEIGHT, IMG_WIDTH))

            img2 = cv2.imread(args.imageB, cv2.IMREAD_COLOR)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.resize(gray2, (IMG_HEIGHT, IMG_WIDTH))
            gray = gray.reshape((IMG_HEIGHT, IMG_WIDTH,1))
            gray2 = gray2.reshape((IMG_HEIGHT, IMG_WIDTH,1))

            inputArray = np.concatenate((gray, gray2), axis=2)
            inputArray = inputArray.reshape(1,IMG_HEIGHT, IMG_WIDTH,2)
            print(inputArray.shape)
            outputTensors = sess.run(regression_node, feed_dict={X:inputArray.astype(np.float32)})
            print("OUTPUT", outputTensors)
        os._exit(0)
    dataset_files = glob.glob(args.img_path + "\\*")
    x_train, x_test = train_test_split(dataset_files, test_size=0.2)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.map(lambda filename: tuple(tf.py_func(create_y, [filename], [tf.float32, tf.float32])))
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors()) 
    

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.map(lambda filename: tuple(tf.py_func(create_y, [filename], [tf.float32, tf.float32])))
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors()) 

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()
    
    number_epochs = 20

    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(test_dataset)
    init_op = tf.global_variables_initializer()

    train_writer = tf.summary.FileWriter(args.output_path + '/log/train')
    test_writer = tf.summary.FileWriter(args.output_path + '/log/test')

    tf.summary.scalar('loss_ss', loss_op)
    merge_op = tf.summary.merge_all()
    sum_i = 0
    sum_b = 0
    loss_ep = 0.70
    saver = tf.train.Saver()

    with tf.Session() as sess:    
        sess.run(init_op)

        for i in range(number_epochs):
            print("EPOCH {} ...".format(i+1))
            
            #Initialize the iterator to consume training data

            sess.run(training_init_op)
            
            accuracy_v = 0.0
            tot_loss = 0
            n_batches = 0
            while True:
                try:                
                    batch_x, batch_y = sess.run(next_element)
                    _, loss_value, summary = sess.run([training_op, loss_op, merge_op],
                     feed_dict={X: batch_x, y:batch_y})
                    print(loss_value, accuracy_v)
                    train_writer.add_summary(summary, sum_i)
                    sum_i += batch_x.shape[0]


                except tf.errors.OutOfRangeError:
                    break
            sess.run(validation_init_op)
            loss_total= 0
            cnt =0
            while True:
                try:                
                    batch_x, batch_y = sess.run(next_element)
                    loss_value, summary = sess.run([loss_op, merge_op],
                     feed_dict={X: batch_x, y:batch_y})
                    print(loss_value, accuracy_v)
                    test_writer.add_summary(summary, sum_b)
                    sum_b += batch_x.shape[0]
                    loss_total +=loss_value
                    cnt +=1
                    
                except tf.errors.OutOfRangeError:
                    break
            if loss_total/float(cnt)<loss_ep:
                loss_ep = loss_total
                save_path = saver.save(sess, args.output_path+ "model"+str(loss_ep)+".ckpt")

                
