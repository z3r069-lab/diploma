from tkinter import*
from tkinter import messagebox
import cv2
import os
import numpy as np

class Train:
    def __init__(self, root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Train Data")
        
    
    def train_classifier(self):
        data_dir=("data")
        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[]
        
        for image in path:
            img=Image.open(image).convert('L')
            ImageNp = np.array(img,'unit8')
            id=int(os.path.split(image)[1].split("."[1]))
            
            faces.append(ImageNp)
            ids.append(id)
            cv2.imshow("Training",ImageNp)
            cv2.waitKey(1)==13
        ids=np.array(ids)
        
        #================= train and save=============
        
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("calssifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("success", "Тренировка завершена")    