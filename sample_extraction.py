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
target_dir = './authorized_person/'
#每個人抓取的臉部樣本數
sample_count = 300
# 互動介面，video裡面的人全都要就輸入 -1 ，只需要某個人的就輸入檔名。
operation = str(input("Welcome to face record program. \nEnter -1 if you want to get landsmark from all films in /video.\nEnter filename (without .mp4) if you want to get landsmark from only one film.\n"))
video_dir_mp4_file = []
if operation == "-1" :
    #尋找Video資料夾裡面的所有影片的檔案(.mp4)
    video_dir_mp4_file = glob.glob("./video/*.mp4")
else :#個人
    operation = "./video\\" + operation + ".mp4"
    video_dir_mp4_file.append(operation)

for personal_video in video_dir_mp4_file:
    start = time.clock()
    #開啟影片，如果是0則是開啟視訊鏡頭
    cap = cv2.VideoCapture(personal_video)
    
    personal_video = personal_video.replace("./video\\", "").replace(".mp4", "")
    #personal_video = personal_video.replace(".mp4", "")
    print("Name : " + personal_video)
    directory = target_dir+personal_video+'/'
    #os.makedirs(directory)

    temp_data = []
    breaked = False
    while (True):
        # ret boolean ,判斷是否有擷取到影像
        ret, frame = cap.read()
        if(ret):
            #降低解析度
            frame = cv2.resize(frame, None,fx=0.5, fy=0.5)
            #
            # 圖片灰階(辨識時會比較快)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scores代表辨識分數，分數越高則人臉辨識的精確率越高，而idx代表臉部方向
            # 第三個參數是指定分數的門檻值，所有分數超過這個門檻值的偵測結果都會被輸出
            faces, scores, idx = detector.run(gray, 0, 0.3)

            for i, d in enumerate(faces) :
                #臉部校正，將臉部圖片翻轉成正面，提高準確度。
                Aligned_face = fa.align(frame, gray, d)
                #縮小圖片
                Aligned_face = cv2.resize(Aligned_face, None,fx=0.5, fy=0.5)
                #
                Aligned_faces, Alig_scores, Alig_idx = detector.run(Aligned_face, 0, 0)
                for alig_i, alig_d in enumerate(Aligned_faces):
                    #抓校正後的人臉
                    Aligned_face_shape = predictor(Aligned_face, alig_d)
                    #np格式轉換(人臉的位置)
                    draw_shape = face_utils.shape_to_np(Aligned_face_shape)
                    for (x,y) in draw_shape :
                        #點出68個臉部特徵點
                        cv2.circle(Aligned_face,(x,y),1,(0,0,255),-1)
                    #計算68個臉部特徵點的歐式距離
                    face_descriptor = np.array(
                        [facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
                    if len(temp_data) == 0:
                        temp_data = face_descriptor
                    else:
                        temp_data = np.append(temp_data, face_descriptor, axis=0)
                #UI介面，進度條
                proccess_percent(len(temp_data),sample_count)
                #顯示圖片(不需顯示可以註解掉)
                cv2.imshow("FaceShow", gray)
                cv2.imshow("aligner",Aligned_face)
            #若樣本數到達我們指定的數量，則儲存成.pkl檔
            if len(temp_data) >= sample_count:
                #開啟資料夾
                os.makedirs(directory)
                #存檔
                joblib.dump(temp_data, directory+'/face_descriptor.pkl')
                print("Finish personal video " + personal_video + " sample extraction .")
                breaked = True
                break



        if cv2.waitKey(1) & breaked :
            break
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    #關閉視窗
    cap.release()
    cv2.destroyAllWindows()
