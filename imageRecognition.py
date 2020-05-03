import cv2 as cv
import sklearn
import numpy
import math
from sklearn import svm
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import random
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

amountOfImages = 0

SIZE = (32,32)

DATASET_PATH = "./dataSet/FIDS30/"
TRAIN_LABEL = ["acerolas","apples","apricots","avocados","bananas","blackberries","blueberries","cantaloupes",
                "cherries","coconuts","figs","grapefruits","grapes","guava","kiwifruit","lemons","limes",
                "mangos","olives","oranges","passionfruit","peaches","pears","pineapples","plums","pomegranates",
                "raspberries","strawberries","tomatoes","watermelons"]

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = cv.imread(image_path)
    image = cv.resize(image,(128,128))
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = numpy.concatenate([dsc, numpy.zeros(needed_size - dsc.size)])
    except cv.error as e:
        print 'Error: ', e
        return None

    return dsc



def imagePreprocessor(path,label):
    amountOfImages = 0
    trainingSet = []
    trainingLabels = []
    for item in label:
        print(item)
        location = path + item + "/"
        for filename in os.listdir(location):
            print(filename)
            #image = extract_features(location + filename)
            image = cv.imread(location + filename)
            image = histogramOfGradientsFeature(image)
            #image = calculateFeatureVector(image)
            trainingSet.append(image)
            trainingLabels.append(item)
            amountOfImages += 1

    return (trainingSet,trainingLabels)


def randomizeOrder(training,label):
    newTraining = []
    newLabel = []
    iterations = len(training)
    for i in range(0,iterations):
        randomIndex = random.randrange(0,len(training))
        newTraining.append(training[randomIndex])
        newLabel.append(label[randomIndex])
        del training[randomIndex]
        del label[randomIndex]
    return (newTraining,newLabel)

def calculateFeatureVector(image,size = SIZE):
    return cv.resize(image,size).flatten()

def prepareSingleImage(imageLocation):
    img = cv.imread(imageLocation)
    img = histogramOfGradientsFeature(img)
    img = img.reshape(1,-1)
    return img

def histogramOfGradientsFeature(image):
    image = cv.resize(image,(64,128))
    image = numpy.float32(image)

    gx = cv.Sobel(image,cv.CV_32F,1,0,ksize=1)
    gy = cv.Sobel(image,cv.CV_32F,0,1,ksize=1)

    mag,angle = cv.cartToPolar(gx,gy,angleInDegrees=True)
    # We bring the angle range from [0,360] to [0,180]
    # since negative parts of the gradient have the same value.
    angle = angle % 180

    i = 0
    histogramOfGradients = []
    for eightByEight in angle:
        histogramOfGradient = [0,0,0,0,0,0,0,0,0]
        k=0
        for pixel in eightByEight:
            # Find the biggest angle of 3 color values in the pixel
            biggestIndex = 0
            if pixel[1] > pixel[biggestIndex]:
                biggestIndex = 1
            if pixel[2] > pixel[biggestIndex]:
                biggestIndex = 2

            # Now we have the biggest angle, find the
            # magnitude corresponding to the biggest angle index
            biggestAngle = pixel[biggestIndex]
            biggestMag = mag[i][k][biggestIndex]
            # find the range into which the angle falls
            normalizedAngle = biggestAngle / 20.0
            foundRange = (int(math.floor(normalizedAngle)),int(math.ceil(normalizedAngle) % 9))

            # Calculate the distance of the angle to upper and lower range
            # and add the magnitude to upper and lower range with a weight
            # respected to how close the angle is to each point(upper,lower)
            if (foundRange[0] == foundRange[1]):
                histogramOfGradient[foundRange[0]] += biggestMag
            elif (foundRange[0] < foundRange[1]):
                factor1 = biggestAngle - foundRange[0]*20.0
                factor2 = foundRange[1]*20.0 - biggestAngle

                histogramOfGradient[foundRange[0]] += factor2*(biggestAngle/(factor1+factor2))
                histogramOfGradient[foundRange[1]] += factor1*(biggestAngle/(factor1+factor2))
            else:
                factor1 = biggestAngle - 160
                factor2 = 180 - biggestAngle

                histogramOfGradient[foundRange[0]] += factor2*(biggestAngle/(factor1+factor2))
                histogramOfGradient[foundRange[1]] += factor1*(biggestAngle/(factor1+factor2))

            k += 1

        histogramOfGradients.append(histogramOfGradient)
        i += 1

    # 16x16 block normalization, each 8x8 is represented by 9x1
    # histograms, 4 8x8 pixels form a 16x16 block, therefore we
    # concat 4 9x1 histograms to get a 36x1 vector, and get
    # l2 norm of that vector to divide each element in the vector
    # to that norm to normalize the values (i.e bring them to 0-1 range)
    normalized = []

    for i in range(0,15):
        startIndex = i*8
        for k in range(0,7):
            # Concatenated array becomes 36x1
            concatArray = numpy.concatenate((histogramOfGradients[startIndex+k],histogramOfGradients[startIndex+k+1],histogramOfGradients[startIndex+k+8],histogramOfGradients[startIndex+k+9]),axis=0)

            sumOfSquares = 0
            for p in concatArray:
                sumOfSquares += p**2

            dividend = math.sqrt(sumOfSquares) # calculate the dividend

            # Divide each value to dividend to normalize the vector
            if dividend != 0:
                for p in range(0,len(concatArray)):
                    concatArray[p] = concatArray[p] / dividend

            normalized.append(concatArray)


    # concatenate all of the 36x1 vectors into one giant vector
    finalFeatureVector = []
    for vector in normalized:
        finalFeatureVector = numpy.concatenate((finalFeatureVector,vector),axis=0)

    return finalFeatureVector


# Support Vector Classification
def supportVectorClassifier(data,target):
    clf = svm.SVC(gamma=0.001,C=100.)
    clf.fit(data,target)
    return clf

# Decision Tree Classification
def decisionTreeClassifier(data,target):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(data,target)
    return clf

def reshaper(old):
    size = len(old)
    new = old.reshape(size,-1)
    return new


def plot_cm(cmList,clfNameList):
    f, axarr = plt.subplots(1,len(cmList))
    for cmIndex in range(0,len(cmList)):
        print(cmIndex)
        img = axarr[cmIndex].matshow(cmList[cmIndex])
        axarr[cmIndex].set_title(clfNameList[cmIndex])
        axarr[cmIndex].set_ylabel('True label')
        axarr[cmIndex].set_xlabel('Predicted label')
        plt.colorbar(img, ax=axarr[cmIndex])
    plt.show()

def normalize(cm):
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    return cm

# Cross validate and construct confusion matrix
def cvd(clf,data,target,clfName):
    scores = cross_val_score(clf,data,target,cv=10)
    accuracy = scores.mean()
    for i in scores:
        print(i)
    print("Accuracy = ",accuracy," (",clfName,")")

def testMode(trainedModelPath):
    clf = joblib.load(trainedModelPath)
    print "Test mode initiated"
    while True:
        location = raw_input("Enter the location of image to test = ")
        image = cv.imread(location)
        image = histogramOfGradientsFeature(image).reshape(1,-1)
        print clf.predict(image)

if __name__ == '__main__':

    trainingSet,trainingLabels = imagePreprocessor(DATASET_PATH,TRAIN_LABEL)

    print amountOfImages,"images used for classification for",len(TRAIN_LABEL),"different fruits."

    #trainingSet,trainingLabels = randomizeOrder(trainingSet,trainingLabels)
    clf = svm.SVC(gamma=0.001,C=100.)
    cvd(clf,trainingSet,trainingLabels,"SVM")

    # Split some data for confusion confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(trainingSet, trainingLabels, random_state=0)

    # Train with the new data
    svmTrained = supportVectorClassifier(X_train, y_train)

    # Predict the test data
    svmPredicted = svmTrained.predict(X_test)

    # Calculate confusion matrix and normalize them
    svm_cm = normalize(confusion_matrix(y_test,svmPredicted))

    # Plot normalized confusion matrix
    plot_cm((svm_cm,svm_cm),("SVM","SVM"))

    trained = supportVectorClassifier(trainingSet,trainingLabels)
    joblib.dump(trained,"FIDStrained.pkl")
