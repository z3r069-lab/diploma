from tkinter import*
from tkinter import ttk
from tkinter import messagebox
import os 
import csv
from tkinter import filedialog


my_data=[]

class Attendace:
    def __init__(self, root)
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Attendace system")
        
        
        self.var_dep=StringVar()
        self.var_course=StringVar()
        self.var_year=StringVar()
        self.var_group=StringVar()
        self.var_ID=StringVar()
        self.var_name=StringVar()
        self.var_time=StringVar()
        self.var_email=StringVar()
        self.var_date=StringVar()
        self.var_status=StringVar()
        
        
        main_frame=Frame(bd=2,bg="white")
        main_frame.place(x=10,y=55,width=1430,height=600)
        
        
        #left label
        
        Left_frame = LabelFrame(main_frame,bg="white", bd=2,relief=RIDGE, text="детали студента", font=("times new roman",12,"bold"))
        Left_frame.place(x=10,y=10,width=730,height=580)
        
        
         #class student info 
        class_student_frame = LabelFrame(Left_frame,bg="white", bd=2,relief=RIDGE, text="инфо студента", font=("times new roman",12,"bold"))
        class_student_frame.place(x=5,y=10,width=710,height=470)
        
        
        # student ID
        studentID_label=Label(class_student_frame, text="ID Студента:", font=("times new roman",12,"bold"),bg="white")
        studentID_label.grid(row=0,column=0,padx=10,pady=5 ,sticky=W)
        
        self.studentID_entry=ttk.Entry(class_student_frame, textvariable=self.var_ID,width=20,font=("times new roman",12,"bold"))
        self.studentID_entry.   grid(row=0,column=1,padx=10,pady=5, sticky=W)
        
         # student name
        Student_name_label=Label(class_student_frame, text="Имя студента:", font=("times new roman",12,"bold"),bg="white")
        Student_name_label.grid(row=0,column=2,padx=10,pady=5 ,sticky=W)
        
        self.student_name_entry=ttk.Entry(class_student_frame,textvariable=self.var_name, width=20,font=("times new roman",12,"bold"))
        self.student_name_entry.grid(row=0,column=3,padx=10,pady=5, sticky=W)
        
        # email
        student_email_label=Label(class_student_frame, text="Напрвление:", font=("times new roman",12,"bold"),bg="white")
        student_email_label.grid(row=1,column=0,padx=10,pady=5, sticky=W)
        
        self.student_email_entry=ttk.Entry(class_student_frame, width=20,textvariable=self.var_course,font=("times new roman",12,"bold"))
        self.student_email_entry.grid(row=1,column=1,padx=10,pady=5, sticky=W)
        
        # department
        department_label=Label(class_student_frame, text="Отделение:", font=("times new roman",12,"bold"),bg="white")
        department_label.grid(row=1,column=2,padx=10,pady=5, sticky=W)
        
        self.department_entry=ttk.Entry(class_student_frame, width=20,textvariable=self.var_dep,font=("times new roman",12,"bold"))
        self.department_entry.grid(row=1,column=3,padx=10,pady=5, sticky=W)
        
        # date
        date_label=Label(class_student_frame, text="дата:", font=("times new roman",12,"bold"),bg="white")
        date_label.grid(row=2,column=0,padx=10,pady=5, sticky=W)
        
        self.date_entry=ttk.Entry(class_student_frame, width=20,textvariable=self.var_date,font=("times new roman",12,"bold"))
        self.date_entry.grid(row=2,column=1,padx=10,pady=5, sticky=W)
        
        # time
        time_label=Label(class_student_frame, text="время:", font=("times new roman",12,"bold"),bg="white")
        time_label.grid(row=2,column=2,padx=10,pady=5, sticky=W)
        
        self.time_entry=ttk.Entry(class_student_frame, width=20,textvariable=self.var_time,font=("times new roman",12,"bold"))
        self.time_entry.grid(row=2,column=3,padx=10,pady=5, sticky=W)
        
        #status
        student_phone_label=Label(class_student_frame, text="Статус:", font=("times new roman",12,"bold"),bg="white")
        student_phone_label.grid(row=3,column=0,padx=10,pady=5, sticky=W)
        
        self.status=ttk.Combobox(class_student_frame,textvariable=self.var_status, font=("times new roman",12,"bold"),width=25,state="readonly")
        self.status["values"] = ("Статус","Присутствует", "Отсутствует")
        self.status.current(0)
        
        self.status.grid(row=3,column=1, padx=10,pady=5,sticky=W)
        
        
         # buttons frame 
        
        btn_frame = Frame(Left_frame, bd=2, bg="white")
        btn_frame.place(x=5,y=480,width=710,height=35)
        
        import_btn= Button(btn_frame,command=self.importCsv,text="Импорт csv",height=1, width=22)
        import_btn.grid(row=0,column=0,padx=5,pady=5)
        
        export_btn= Button(btn_frame,command=self.exportCsv, text="Экпорт csv",height=1, width=22)
        export_btn.grid(row=0,column=1,padx=5,pady=5)
        
        #update_btn= Button(btn_frame, text="Обновить",height=1, width=22)
        #update_btn.grid(row=0,column=2,padx=5,pady=5)
        
        reset_btn= Button(btn_frame, command=self.reset_data,text="Перезагрузить",height=1, width=22)
        reset_btn.grid(row=0,column=3,padx=5,pady=5)
        
        
        
         #right label
        
        Right_frame = LabelFrame(main_frame,bg="white", bd=2,relief=RIDGE, text="детали студента", font=("times new roman",12,"bold"))
        Right_frame.place(x=750,y=10,width=660,height=580)
        
        
        table_frame = LabelFrame(Right_frame,bg="white", bd=2,relief=RIDGE)
        table_frame.place(x=5,y=10,width=640,height=543)
        
        scroll_x=ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scroll_y=ttk.Scrollbar(table_frame,orient=VERTICAL)
        
        self.student_table=ttk.Treeview(table_frame,columns=("ID","Name","dep","course","date","time","status"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)
        
        scroll_x.pack(side=BOTTOM,fill=X)
        scroll_y.pack(side=RIGHT,fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)
        
        
        self.student_table.heading("ID",text="ID")
        self.student_table.heading("Name",text="имя")
        self.student_table.heading("dep",text="Отделение")
        self.student_table.heading("course",text="Направление")
        self.student_table.heading("date",text="Дата")
        self.student_table.heading("time",text="Время")
        self.student_table.heading("status",text="Статус")
        self.student_table["show"]="headings"
        
        
        self.student_table.column("ID",width=50)
        self.student_table.column("Name",width=150)
        self.student_table.column("dep",width=300)
        self.student_table.column("course",width=300)
        self.student_table.column("time",width=100)
        self.student_table.column("date",width=100)
        self.student_table.column("status",width=130)
              
        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        
        self.student_table.pack(fill=BOTH,expand=1)
        
    
    
    
    
    
        
    def fetch_data(self,rows):
        self.student_table.delete(*self.student_table.get_children())
        for i in rows:
            self.student_table.insert("",END,values=i)
    
    
    #import csv
    def importCsv(self):
        global my_data
        my_data.clear()
        fln= "attendance/attendance.csv"
        #fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Open CSV",filetypes=(("CSV File","*csv"),("All Files","*.*")),parent=self.root)  
        with open(fln) as myFile:
            csvread=csv.reader(myFile,delimiter=",")
            for i in csvread:
                my_data.append(i)
            self.fetch_data(my_data)      
    
    
    def exportCsv(self):
        try:
            if len(my_data)<1:
                messagebox.showerror("No Data","данные не найдены",parent=self.root)
                return False
            fln=filedialog.asksaveasfilename(initialdir=os.getcwd(),title="Open CSV",filetypes=(("CSV File","*csv"),("All Files","*.*")),parent=self.root)  
            with open(fln,mode="w",newline="") as myFile:
                exp_write=csv.writer(myFile,delimiter=",")
                for i in my_data:
                    exp_write.writerow(i)
                messagebox.showinfo("Success", "Сохранено в "+os.path.basename(fln),parent=self.root)    
        except Exception as es:
                messagebox.showerror("Fail",f"Из за ошибки:{str(es)}", parent=self.root)    

    def get_cursor(self,event=""):
        cursor_focus=self.student_table.focus()
        content=self.student_table.item(cursor_focus)
        data=content["values"]
        
        self.var_ID.set(data[0]),
        self.var_name.set(data[1]),
        self.var_dep.set(data[2]),
        self.var_course.set(data[3])
        self.var_time.set(data[4]),
        self.var_date.set(data[5]),
        self.var_status.set(data[6])
        
    def reset_data(self):
        
        self.var_time.set("")
        self.var_date.set("")    
        self.var_course.set("")
        self.var_dep.set("")
        self.var_ID.set("")
        self.var_name.set("")
        self.var_status.set("Статус")
        
       
        
if __name__ == "__main__":
    root=Tk()
    obj=Attendace(root)
    root.mainloop()
