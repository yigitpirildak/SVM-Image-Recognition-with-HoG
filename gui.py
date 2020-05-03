from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import ImageTk,Image
import tester
import imageRecognition

def chooseImage():
    filename = askopenfilename()
    widget.img = ImageTk.PhotoImage(Image.open(filename).resize((200,200)))
    widget['image'] = widget.img
    imgPrepared = imageRecognition.prepareSingleImage(filename)
    prediction = clf.predict(imgPrepared)
    predictionText.set(prediction)
    return filename


if __name__ == '__main__':
    # Initialize the classifier.
    clf = imageRecognition.joblib.load(tester.MODEL) # Load the pretrained model

    root = Tk()

    b = Button(root, text="Choose Image", command=chooseImage)

    widget = Label(root, compound='top')
    widget.img = ImageTk.PhotoImage(Image.open("./mytest/apple3.png").resize((200,200)))
    widget['image'] = widget.img

    predictionText = StringVar()
    predictionTextLabel = Label(root,textvariable=predictionText,relief=RAISED)
    imgPrepared = imageRecognition.prepareSingleImage("./mytest/apple3.png")
    prediction = clf.predict(imgPrepared)
    predictionText.set(prediction)

    b.pack()
    widget.pack()
    predictionTextLabel.pack()

    mainloop()
