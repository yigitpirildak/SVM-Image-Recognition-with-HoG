from imageRecognition import imagePreprocessor as ip
from sklearn.externals import joblib

TEST_PATH = "./dataSet/Test/"

TRAIN_LABEL = ["apples","apricots","avocados","bananas",
                "cherries","coconuts","grapes","guava","kiwifruit","lemons","limes",
                "mangos","oranges","passionfruit","peaches","pears","pineapples","plums","pomegranates",
                "raspberries","strawberries"]

MODEL = "./FIDStrained.pkl"



if __name__ == '__main__':
    truePrediction = 0
    falsePrediction = 0

    dataSet,labels = ip(TEST_PATH,TRAIN_LABEL) # Load the test images
    clf = joblib.load(MODEL) # Load the pretrained model

    for i in range(0,len(dataSet)):
        prediction = clf.predict(dataSet[i].reshape(1,-1))
        print(prediction)
        if prediction == labels[i]:
            truePrediction += 1
        else :
            falsePrediction += 1

    print "True predictions =",truePrediction
    print "False prediction =",falsePrediction
    print "Accuracy = %",(truePrediction*100)/(truePrediction+falsePrediction)
