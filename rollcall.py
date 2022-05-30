import dlib
import cv2
import os
import csv
from tensorflow.python.keras.models import model_from_json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import time
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils
import time
ROOT_DIR = os.getcwd()
#輸入當日日期
datename = input("Date?")
#資料位置設定
predictor_path = './face_detect_landmarks_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './face_detect_landmarks_model/dlib_face_recognition_resnet_model_v1.dat'
hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'
MODEL_PATH = os.path.join(ROOT_DIR, "model")
nn_model_dir = os.path.join(ROOT_DIR, "nn_model")
nn_model_dir = nn_model_dir + "\\"

#資料載入
json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()
#模型載入
dnn_model = model_from_json(json_model)
dnn_model.load_weights(nn_model_dir+hdf5_filename)
label_dict=joblib.load(nn_model_dir+labeldict_filename)

# 人臉辨識
detector = dlib.get_frontal_face_detector()
# 人臉特徵點辨識
predictor = dlib.shape_predictor(predictor_path)
# 人臉校正
fa = FaceAligner(predictor)
# 將人臉的資訊提取成一個128维的向量，如果臉部更相似，距離會更加接近、符合
# 使用歐幾里得距離來計算，公式:distence = sqrt((x1-x2)^2+(y1-y2)^2)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
dnn_model = model_from_json(json_model)
dnn_model.load_weights(nn_model_dir+hdf5_filename)


def face_predict(faces, gray, frame) :
    for d, face in enumerate(faces) :
        #抓取臉部位置
        x = face.left()
        y = face.top()
        w = face.right()-face.left()
        h = face.bottom() - face.top()

        identity = ""
        #將臉部校正後進行臉部抓取
        Aligned_face = fa.align(frame, gray, face)
        Aligned_face = cv2.resize(Aligned_face, None,fx=0.5, fy=0.5)
        Aligned_faces, Alig_scores, Alig_idx = detector.run(
                Aligned_face, 0, 0)

        for alig_i, alig_d in enumerate(Aligned_faces):
                #抓取臉部特徵點並計算歐式距離
                Aligned_face_shape = predictor(Aligned_face, alig_d)        
                face_descriptor = np.array(
                            [facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
                #進行預測
                prediction = dnn_model.predict(face_descriptor)
                #得到信心度
                max_prob = np.max(prediction)
                #得到信心度最高的index
                index = np.argmax(prediction)
                #得到信心度最高的學號
                identity = label_dict[index]
                if identity == label_dict[len(label_dict) - 1] :
                    #偵測成未知，用紅色方框框起
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                else :
                    #偵測程某個同學，用綠色方框框起
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                #在圖片上寫上學號以及信心度
                cv2.putText(frame,str(identity) + " (" + str(max_prob) + ")",(x+5,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 2)
        return identity


if __name__ == '__main__':
    #開啟視訊鏡頭
    cap = cv2.VideoCapture(0)

    frame_count = 0
    current_identity = ""
    identity_list = []
    while(True) :
        #讀取畫面
        ret, frame = cap.read()
        #轉換灰階集偵測臉部照片
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, scores, idx = detector.run(gray, 0, 0)
        #若有偵測到，scores會有內容，沒有則無
        if len(scores) > 0 :
            #進行預測並回傳結果
            identity = face_predict(faces,gray,frame)
            #若有讀到照片且預測的人不是未知的話 進行點名
            if str(identity) != "" and str(identity) != "UNKNOWN" :
                #若這個人已經點過
                if str(identity) in identity_list :
                    print("你已經點過囉!")
                #這個人沒點過且這個人是目前正在點的人
                elif str(identity) == current_identity :
                    frame_count = frame_count + 1
                    if frame_count >= 20 :
                        print("點名成功")
                        frame_count = 0 
                        #把點到的人加入點名清單
                        identity_list.append(str(identity))
                
                else :#不是目前正在點的人
                    frame_count = 1
                    current_identity = str(identity)
        #顯示視窗，在視窗內按下q則結束程式
        cv2.imshow("FaceShow", frame)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    #生成點名檔(csv)
    fp = open("./output/" + datename + ".csv","a")
    for name in label_dict :
        #若此類別不是未知且這個類別有在剛剛的點名清單
        if label_dict[name] != "UNKNOWN" and str(label_dict[name]) in identity_list :
            #登記成有到
            fp.write(str(label_dict[name]) + ",1\n" )
        #若此類別不是未知且這個類別沒有在剛剛的點名清單
        elif label_dict[name] != "UNKNOWN" and not (str(label_dict[name]) in identity_list) :
            #登記成未到
            fp.write(str(label_dict[name]) + ",0\n")
    fp.close()
        
    

    