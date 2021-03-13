# Import OpenCV2 for image processing
import cv2,os

# Import numpy for matrices calculations
import numpy as np

import mysql.connector

#connect database
def connectdb():
    print('连接到mysql服务器...')
    # 打开数据库连接
    # 用户名:hp, 密码:Hp12345.,用户名和密码需要改成你自己的mysql用户名和密码，并且要创建数据库TESTDB，并在TESTDB数据库中创建好表Student
    db = mysql.connector.connect(user="root", passwd="mysql", database="vipclub", use_unicode=True)
    print('连接上了!')
    return db

#select
def querydb(db,id):
    # 使用cursor()方法获取操作游标 
    cursor = db.cursor()
    print(id)
    # SQL 查询语句
    sql = 'SELECT * FROM consummer where user_id= %s' % (id)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
       
        if results==():
            print("NULL")
            msg="Unknown"
        else: 
            for row in results:
               user_name=row[1]
               user_sex=row[2]
               user_phone=row[3]
               user_createtime=row[4]
               user_age=row[5]
            # 打印结果
            if user_sex == 0:
                sex='F'
            else:
                sex='M'
            msg = 'ID: %s\n Name: %s\n sex: %s\n phone: %s\n createtime: %s\n age: %s' % \
                   (id,user_name, sex,user_phone,user_createtime,user_age)
    except:
        msg = 'Error: unable to fecth data'
    return msg


current_dir = os.path.dirname(os.path.abspath(__file__))
print('current_dir='+current_dir)

search_button=False
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read(current_dir+"//trainer/trainer.yml")

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

db=connectdb()
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    key = cv2.waitKey(33) & 0xFF
    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        
        if key==ord('s'):
            #查询资料，并回显
            search_button = not search_button
            msg=querydb(db,Id[0])


          
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, "Morning^_^", (x,y-40), font, 2, (255,255,255), 3)
        if search_button:
            y0, dy = 50, 25 
            for i, txt in enumerate(msg.split('\n')):
                y = y0+i*dy
                cv2.putText(im,str(txt),(50, y), font, 1,(255,255,255), 1)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

db.close();
# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
