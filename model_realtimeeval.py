from ultralytics import YOLO
import os
from  pathlib import Path
import cv2
# This is a memory sensitive task, so CLOSE YOUR BROWSER before running this script!!!!!!
# your browser can occupy more than 10 GB memory, so remember to close it before you run this script!
# you can press q to exit

#if you choose a video then it says: [ WARN:0@10.870] global cap_msmf.cpp:936 CvCapture_MSMF::initStream Failed to set mediaType (stream 0, (3840x2160 @ 29.97) MFVideoFormat_RGB32(codec not found)
#then you have to close the terminal. This is because Opencv don't support rgb32 and 4k res video..........

tank_detect_model = "tanks"
drone_detect_model = "drones"

def realtime_eval_og(model_type, url):
    currentdir = Path(".")
    modellocation = currentdir / "models" / model_type / "best.pt"

    model = YOLO(modellocation)  

    online_url = url
    # 直接预测视频文件
    results = model.predict(
        source=online_url, 
        show=True,#show the video
        conf=0.5, # set a smaller value if you want to detected approximately. the smaller the value, the more possibility for bounding tanks
        save=False,
        vid_stride=2, #common out this line if you have 16GB ram
        stream_buffer=False #common out this line if you have 16GB ram

    )

def realtime_eval_cv(model_type, url):
    currentdir = Path(".")
    modellocation = currentdir / "models" / model_type / "best.pt"

    model = YOLO(modellocation)  
    online_url = url
    results = model.predict(
        source=online_url,
        stream=True,
        conf=0.4, ## set a smaller value if you want to detected approximately. the smaller the value, the more possibility for bounding tanks
        save=False,
    )

    for result in results:
        frame = result.plot()
        cv2.imshow("Tank Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    proxy_url = "http://127.0.0.1:7897"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

    video_url = ["https://youtu.be/sj5ufK4epmA",  \
        #this is tank video
        "https://youtu.be/ambp_X7m-2s",\
        #tanks video 2
        "https://youtu.be/1U7Bu2pSUwU",\
        #this is heli video
                 ]

#    realtime_eval_og()

    realtime_eval_cv(tank_detect_model, video_url[1])
