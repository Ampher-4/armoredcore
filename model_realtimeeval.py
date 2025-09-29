from ultralytics import YOLO
import os
from  pathlib import Path
import cv2
# This is a memory sensitive task, so CLOSE YOUR BROWSER before running this script!!!!!!
# your browser can occupy more than 10 GB memory, so remember to close it before you run this script!

def realtime_eval_og():
    currentdir = Path(".")
    modellocation = currentdir / "models" / "tanks" / "best.pt"

    model = YOLO(modellocation)  

    online_url = "https://youtu.be/sj5ufK4epmA"
    # 直接预测视频文件
    results = model.predict(
        source=online_url, 
        show=True,#show the video
        conf=0.5, # set a smaller value if you want to detected approximately. the smaller the value, the more possibility for bounding tanks
        save=False,
        vid_stride=2, #common out this line if you have 16GB ram
        stream_buffer=False #common out this line if you have 16GB ram

    )

def realtime_eval_cv():
    currentdir = Path(".")
    modellocation = currentdir / "models" / "tanks" / "best.pt"

    model = YOLO(modellocation)  
    online_url = "https://youtu.be/sj5ufK4epmA"
    results = model.predict(
        source=online_url,
        stream=True,
        conf=0.5,
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

#    realtime_eval_og()
    realtime_eval_cv()

