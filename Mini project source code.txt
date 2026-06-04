from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.applications import VGG19
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
import pickle
import os
from keras.models import load_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pydicom as dicom

main = tkinter.Tk()
main.title("Alzheimer Brain Tumor Detection using VGG16 & 19 Algorithms")
main.geometry("1300x1200")

global filename
global classifier
global labels, X, Y, X_train, y_train, X_test, y_test, vgg16_model

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    labels = ['Normal', 'Alzheimer Brain Tumor']
    text.insert(END,"Tumor found in dataset are\n\n")
    for i in range(len(labels)):
        text.insert(END,labels[i]+"\n")

def processDataset():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    ds = dicom.dcmread(root+'/'+directory[j])
                    img = ds.pixel_array
                    cv2.imwrite("test.png", img*255)
                    img = cv2.imread("test.png")
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    if name == 'Normal':
                        Y.append(0)
                    else:
                        Y.append(1)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    text.insert(END,"Dataset Preprocessing Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% images are used to train VGG16 & 19 : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images are used to test  : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    img = X[0]
    img = cv2.resize(img, (200, 200))
    cv2.imshow("Brain Image", img)
    cv2.waitKey(0)

def trainVGG16():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, labels, vgg16_model
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in vgg16.layers:
        layer.trainable = False
    vgg16_model = Sequential()
    vgg16_model.add(vgg16)
    vgg16_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg16_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(units = 256, activation = 'relu'))
    vgg16_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vgg16_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg16_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg16_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg16_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg16_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg16_model = load_model("model/vgg16_weights.hdf5")
    predict = vgg16_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    text.insert(END,"VGG16 Accuracy  : "+str(a)+"\n")
    text.insert(END,"VGG16 Precision : "+str(p)+"\n")
    text.insert(END,"VGG16 Recall    : "+str(r)+"\n")
    text.insert(END,"VGG16 FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("VGG16 Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainVGG19():
    global filename, X, Y, X_train, y_train, X_test, y_test, labels, vgg19_model
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in vgg19.layers:
        layer.trainable = False
    vgg19_model = Sequential()
    vgg19_model.add(vgg19)
    vgg19_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg19_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg19_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg19_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg19_model.add(Flatten())
    vgg19_model.add(Dense(units = 256, activation = 'relu'))
    vgg19_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vgg19_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg19_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg19_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg19_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg19_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg19_model = load_model("model/vgg19_weights.hdf5")
    predict = vgg19_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    text.insert(END,"VGG19 Accuracy  : "+str(a)+"\n")
    text.insert(END,"VGG19 Precision : "+str(p)+"\n")
    text.insert(END,"VGG19 Recall    : "+str(r)+"\n")
    text.insert(END,"VGG19 FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("VGG19 Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    

def graph():
    f = open('model/vgg16_history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    vgg16_accuracy = graph['val_accuracy']

    f = open('model/vgg19_history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    vgg19_accuracy = graph['val_accuracy']    
    vgg19_accuracy = vgg19_accuracy[10:30]

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(vgg16_accuracy, 'ro-', color = 'green')
    plt.plot(vgg19_accuracy, 'ro-', color = 'blue')
    plt.legend(['VGG16 Accuracy', 'VGG19 Accuracy'], loc='upper left')
    plt.title('VGG16 & 19 Training Accuracy Graph')
    plt.show()
    

def predictTumor():
    global vgg19_model, labels
    filename = filedialog.askopenfilename(initialdir="testData")
    ds = dicom.dcmread(filename)
    img = ds.pixel_array
    cv2.imwrite("test.png", img*255)
    img = cv2.imread('test.png')
    img = cv2.resize(img, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg19_model.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread("test.png")
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Alzhimer Brain Tumor : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Alzhimer Brain Tumor : '+labels[predict], img)
    cv2.waitKey(0)
    

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Alzheimer Brain Tumor Detection using VGG16 & 19 Algorithms',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Dicom Alzheimer Brain Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=200)
processButton.config(font=font1)

trainvggButton = Button(main, text="Train VGG16 Algorithm", command=trainVGG16)
trainvggButton.place(x=50,y=250)
trainvggButton.config(font=font1)

vggButton = Button(main, text="Train VGG19 Algorithm", command=trainVGG19)
vggButton.place(x=50,y=300)
vggButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Tumor Detection", command=predictTumor)
predictButton.place(x=50,y=400)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=370,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
