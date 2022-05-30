這是許嘉榮、劉信儀與莊凱丞於 2018/11 完成的中原大學畢業專題。

動機 :

在大學上課期間，發現常常在點名的時候會出現代點的現象。但教授一個人不可能認得
那麼多張臉，除了檢舉很難發現這個代點的現象。若透過視訊鏡頭來做臉部辨識，搭配
機器學習，就能解決代點的問題。也不需要透過傳統教授唱名的方式一個一個點名，耗
時又費力。可架設在教室門口，學生在進教室時就能自己點名，有效的降低代點以及點
名的時間。

預計執行方式 :

蒐集同學的臉部影片(大約三十秒)，透過程式擷取同學們的臉部特徵點進行訓練，得到臉
部辨識的模組，再藉由 Webcam 來辨識同學們的臉進行點名，點名完會輸出一個csv檔，匯入
學校的點名系統，以完成點名。

前置作業 :

  到 http://dlib.net/files/ 下載 shape_predictor_68_face_landmarks.dat 及 dlib_face_recognition_resnet_model_v1.dat 並放入 face_detect_landmarks_model 資料夾內


實際執行方式 : 

1. sample_extraction.py (藉由影片抓取臉部特徵)
    將影片都放入名為 video 的資料夾，並為每個影片取上名字(未來會藉由此名字當作學號輸出)，檔案格式須為.mp4檔
    執行python sample_extraction.py 互動介面會告訴你若要在資料夾內的影片全都進行抓取則輸入 -1 
    若要抓取某個人的特徵則輸入檔案名稱(去除.mp4的檔案名稱)
    抓取結束後會將結果輸出至名為authorized_person的資料夾內，依照學號排序。

2. unknown_face_record.py (藉由圖片抓取未知人臉特徵)
    將大量人臉圖片放入名為 unknown_face 的資料夾，圖片須為.jpg檔
    執行python unknown_face_record.py 互動介面會通知你要輸入 -1 來抓取未知人臉。
    其中使用到shuffle來亂數選取未知人臉而非照著檔案名稱進行選取。
    抓取結束後會將結果輸出至 unknown_person/preprocessed_data 內。

3. model_train.py (模型訓練)
    可至程式內調整各項深度神經網路的超參數。
    直接執行 python model_train.py 來訓練模型，訓練完後會輸出至名為 nn_model 的資料夾內。

4. rollcall.py (點名程式)
    輸入 python rollcall.py 後，執行介面會問你今日的日期，輸入日期後(例: 0523 )會在output這個資料夾內建立一個名為日期的csv檔(例: 0523.csv)
    接著會出現一個視訊畫面，若已是點過的同學，他會在cmd介面告訴你已經點過了
    若沒有點過，他會開始讀取，你必須在框框內顯示為同一個同學達到一定張數(偵數)才會通知點名成功。
    若過程中被判斷成是別人則需要重新計算到一定偵數才會點名成功。
    點名結束後可以在是窗內按下q來結束程式，並會輸出剛剛的點名結果至指定資料夾完成點名。


安裝時會遇到的問題 :

1. dlib 安裝問題 
    (1) 可能沒有安裝Visual Studio導致，確定已經安裝並安裝了cmake for windows tool 
    (2) 未安裝cmake或cmake未放入環境變數內，確保上述兩情況有達成。


執行時遇到的問題 :

1. unable to open XXX
    可以試看看路徑都不要有中文出現。

2. TypeError: Can't parse 'center'. Sequence item with index 0 has a wrong type
    將 facealigner.py 裡面的 第64與65行 
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    改成 eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)
    並儲存。進入facealigner.py的方式可以在錯誤碼出現時，直接按ctrl+左鍵就可以進入並修改。

3. ....... in load dispatch[key[0]](self) KeyError: 0
    joblib版本問題，若你是使用 from sklearn.externals import joblib，就要確保無論是訓練或是執行的檔案都必須是由from sklearn.externals import joblib所執行出來的結果，否則會衝突並出現這樣的錯誤。

4. FileExistsError: [WinError 183] 當檔案已存在時，無法建立該檔案。
    在執行sample_extraction.py前必須確保要抓取臉部特徵的影片沒有檔案或資料夾已經存在的情況。

