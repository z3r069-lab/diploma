import tkinter as tk
import subprocess
from tkinter import ttk
from student import Student
from attendance import Attendace
import os
from datetime import datetime
import mysql.connector
from PIL import Image
import numpy as np
import cv2
from tkinter import messagebox
import codecs

class Face_recognition_system:
    def __init__(self, root):
        self.root=root
        self.root.geometry("660x480+0+0")
        self.root.title("face recognition system")
        
        
        #student button
        b1 = tk.Button(text="детали студента", command=self.student_details ,cursor="hand2")
        b1.place(x=100,y=100,width=220,height=60)
        
        #face detection
        b2 = tk.Button(text="сканировать лицо", command=self.face_recodnition,cursor="hand2")
        b2.place(x=100,y=170,width=220,height=60)
        
        #attendance registration
        b3 = tk.Button(text="ометить посещаемость",command=self.open_attendance, cursor="hand2")
        b3.place(x=100,y=240,width=220,height=60)
        
        #help button
        b4 = tk.Button(text="помощь",command=self.help, cursor="hand2")
        b4.place(x=100,y=320,width=220,height=60)
        
        #train data button
        b5 = tk.Button(text="тренировать данные",command=self.train_classifier, cursor="hand2")
        b5.place(x=330,y=100,width=220,height=60)
        
        #photos
        b6 = tk.Button(text="фото", command=self.open_img,cursor="hand2")
        b6.place(x=330,y=170,width=220,height=60)
        
        #dev
        b7 = tk.Button(text="разработчик", cursor="hand2")
        b7.place(x=330,y=240,width=220,height=60)
        
        #exit button
        b8 = tk.Button(text="выход", command=self.Exit,cursor="hand2")
        b8.place(x=330,y=320,width=220,height=60)
        
    
    #===============function buttons===============
    
    def help(self):
        # Create a new window for help
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        
        # Create a Text widget for displaying the help text
        help_text = tk.Text(help_window, width=80, height=20)
        help_text.pack(fill=tk.BOTH, expand=True)
        
       
        
        # Read the help text from a file
        file_path = "help.txt"  # Replace with the path to your help file
        with codecs.open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Set the help text in the Text widget
        help_text.insert(tk.END, text)
        
        # Disable editing of the Text widget
    
    
    
    def student_details(self):
        path = "student.py"
        subprocess.run(["python",path])
        
        
          
        
    def open_attendance(self):
        
        path = "attendance.py"
        subprocess.run(["python",path])
        
                
        
    
    def Exit(self):
        self.root.destroy()
    
    #=============== attendance=====================
    
    def mark_attendance(self,i,n,c,d):
        with open("attendance/attendance.csv","r+",newline="\n") as f:
            
            myDataList=f.readlines()
            name_list=[]
            for line in myDataList:
                entry=line.split((","))
                name_list.append(entry[0])
            if((i not in name_list) and (n not in name_list) and (c not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1=now.strftime("%d/%m/%Y")
                dtString=now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{n},{c},{d},{dtString},{d1},Присутствует")      
    
    
    
    
    
    
    
    
    
    
    
    def open_img(self):
        os.startfile("data")
    
    def train_classifier(self):
        data_dir=("data")
        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[]
        
        for image in path:
            img=Image.open(image).convert('L')
            ImageNp = np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])
            
            faces.append(ImageNp)
            ids.append(id)
            cv2.imshow("Training",ImageNp)
            cv2.waitKey(1)==13
        ids=np.array(ids)
        
        #================= train and save=============
        
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("success", "Тренировка завершена")     
    
    
    
    def face_recodnition(self):
        
        file = open("attendance/attendance.csv","r+",newline="\n")
        file.truncate(0)
        file.close
        
        def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf):
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
            
            coord=[]
            
            for(x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h,x:x+w])
                confidence=int((100*(1-predict/300)))
                
                conn=mysql.connector.connect(host="192.168.0.10",port="3308",username="diploma_admin",password="root",database="students")
                my_cursor=conn.cursor()
                
                my_cursor.execute("select Id from students where Id="+str(id))
                i=my_cursor.fetchone()
                i="+".join(i)
                
                my_cursor.execute("select name from students where Id="+str(id))
                n=my_cursor.fetchone()
                n="+".join(n)
                
                my_cursor.execute("select Course from students where Id="+str(id))
                c=my_cursor.fetchone()
                c="+".join(c)
            
                my_cursor.execute("select Dep from students where Id="+str(id))
                d=my_cursor.fetchone()
                d="+".join(d)
                
                if confidence>77:
                    cv2.putText(img,f"ID:{i}",(x,y-70),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                    cv2.putText(img,f"Имя:{n}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                    cv2.putText(img,f"Отделение:{d}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                    cv2.putText(img,f"Курс:{c}",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                    self.mark_attendance(i,n,c,d)
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,f"Unknown",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                
        
                coord=[x,y,w,y]          
                
            return coord                
        
        def recognize(img,clf,faceCascade):
            coord=draw_boundary(img,faceCascade,1.1,10,(255,25,255),"Face",clf)
            return img            
        
        faceCascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")
        
        video_cap=cv2.VideoCapture(0)
        
        while True:
            ret,img=video_cap.read()
            img=recognize(img,clf,faceCascade)
            cv2.imshow("Face recognition",img)
            
            if cv2.waitKey(1)==13:
                break
                
        video_cap.release()
        cv2.destroyAllWindows() #press enter to close
        

        
        
if __name__ == "__main__":
    root=tk.Tk()
    obj=Face_recognition_system(root)
    root.mainloop()
       

        
