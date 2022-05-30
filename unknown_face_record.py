from imutils import face_utils
from imutils.face_utils import FaceAligner
from sklearn.externals import joblib
import numpy as np
import cv2
import dlib
import glob
import os
import sys
import time
import random
def proccess_percent(cur, total) :
    if cur+1 ==total :
        percent = 100.0
        print('Sample Extraction Processing : %5s [%d/%d]'%(str(percent)+'%',cur+1,total),end='\n')
    else :
        percent = round(1.0*cur/total*100,1)
        print('Sample Extraction Processing : %5s [%d/%d]'%(str(percent)+'%',cur+1,total),end='\r')

# DLIB's model path for face pose predictor and deep neural network model
predictor_path = './face_detect_landmarks_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './face_detect_landmarks_model/dlib_face_recognition_resnet_model_v1.dat'

# 人臉辨識
detector = dlib.get_frontal_face_detector()
# 人臉特徵點辨識
predictor = dlib.shape_predictor(predictor_path)
# 人臉校正
fa = FaceAligner(predictor)
# 將人臉的資訊提取成一個128维的向量，如果臉部更相似，距離會更加接近、符合
# 使用歐幾里得距離來計算，公式:distence = sqrt((x1-x2)^2+(y1-y2)^2)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Set the output directory of the user's data

#設定張數
sample_count = 1500
# 未知人像需要放的地方
directory = './unknown_person\preprocessed_data'
operation = str(input("Welcome to face record program. \nEnter -1 if you want to get landsmark from all films in /unknown_face.\n"))

pic_dir_jpg_file = []
if operation == "-1" :
    pic_dir_jpg_file = glob.glob("./unknown_face/*.jpg")
else :
    operation = "./unknown_face\\" + operation + ".jpg"
    pic_dir_jpg_file.append(operation)
start = time.clock()
temp_data = []
# suffle增加隨機性(因unknown_face裡的照片會有某個人連續出現的問題)
random.shuffle(pic_dir_jpg_file)
for personal_pic in pic_dir_jpg_file:
    frame = cv2.imread(personal_pic)
    
    personal_pic = personal_pic.replace("./unknown_face\\", "").replace(".jpg", "") #照片須是jpg檔
    print("Name : " + personal_pic)

    
    breaked = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #將照片進行灰階，偵測較快

    faces, scores, idx = detector.run(gray, 0, 0) 
    if len(scores) > 0 and scores[0] >= 0.1 :
        for i,d in enumerate(faces) :
            Aligned_face = fa.align(frame, gray, d)
            Aligned_faces, Alig_scores, Alig_idx = detector.run(Aligned_face, 0, 0)
            
            for alig_i, alig_d in enumerate(Aligned_faces):
                Aligned_face_shape = predictor(Aligned_face, alig_d)
                draw_shape = face_utils.shape_to_np(Aligned_face_shape)
                for (x,y) in draw_shape :
                    cv2.circle(Aligned_face,(x,y),1,(0,0,255),-1)
                face_descriptor = np.array(
                    [facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
                if len(temp_data) == 0:
                    temp_data = face_descriptor
                else:
                    temp_data = np.append(temp_data, face_descriptor, axis=0)
 
    if len(temp_data) >= sample_count:

        joblib.dump(temp_data, directory+'/face_descriptor.pkl')
        print("Finish personal video " + personal_pic + " sample extraction .")

        breaked = True
        break
    ''' 如有需要將整個影片跑完的話，把註解拿掉,sample_count 值要改大一點
    else :
        os.makedirs(directory)
        joblib.dump(temp_data, directory+'/face_descriptor.pkl')
        print("Finish personal video " + personal_pic + "sample extraction .")
        #print("Sample number = " + format(len(temp_data)))
        breaked = True
        break
    '''

    cv2.imshow("FaceShow", gray)
    cv2.imshow("aligner",Aligned_face)     
    if cv2.waitKey(1) & breaked :
        break
    # time.sleep(0.3)
    print(len(temp_data))
    cv2.destroyAllWindows()

    

elapsed = (time.clock() - start)
print("Time used:",elapsed)
# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
