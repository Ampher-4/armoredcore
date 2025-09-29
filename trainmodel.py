"""
.
├─ dataset/
├─ saves/
├─ model.pt
└─ thisfile!!!.py
"""

import os
import torch
from ultralytics import YOLO
#from ultralytics.utils.callbacks import add_callback

def test_cuda():
    print(torch.__version__)
    print(torch.cuda.is_available())   
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    print(os.path.curdir)
    print(os.path.exists('dronedata.yaml'))
    LOCAL_WEIGHT = 'best.pt'
    model = YOLO(LOCAL_WEIGHT)

    #onnx save callback

    #NOTE: this callback is wrong. see docs for detail. yolo has built-in automatic checkpoint features
    #def onnx_export_callback(trainer):
    #    epoch = trainer.epoch + 1
    #    if epoch % 5 == 0:
    #        onnx_path = SAVE_DIR / f"tank_v8n_epoch{epoch}.onnx"
    #        trainer.model.export(
    #            format='onnx',
    #            imgsz=trainer.args.imgsz,
    #            half=False,
    #            opset=12,
    #            simplify=True,
    #            save_dir=SAVE_DIR,
    #            weights=onnx_path
    #        )
    #        print(f'>>> ONNX saved: {onnx_path}')

    #this api get deprecated!!!!!
    ##add_callback('train', 'on_train_epoch_end', onnx_export_callback)
    #model.add_callback("on_train_epoch_end", onnx_export_callback)


    #actual train code
    #para list docs: https://docs.ultralytics.com/zh/modes/train/#train-settings
    #TODO: add more descriptions on config behaviour, or write tech docs
    try:
        model.train(
            data='dronedata.yaml',#this file must exists under running directory!
            epochs=20,
            imgsz=640,
            batch=32,
            name='TankandHeli_v8n',
            exist_ok=False,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            warmup_epochs=3,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.2,
            device=0,
            amp=False, #we are focusing on real-time performance, so no amp here
            save_period=5 #save every 5 epochs
        )
        model.export(format='onnx', imgsz=640, half=False, simplify=True)
    except KeyboardInterrupt:
    #saves on accident quit
        model.export(format='onnx', imgsz=640, half=False, simplify=True)
        print(f'saved model')


def predict(model, tgt):
    from ultralytics import YOLO
import os
from datetime import datetime



def yolo_predict(model_path, image_path,  use_proxy=True):

    if use_proxy:
        proxy_url = "http://127.0.0.1:7897"
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url




    if not os.path.exists(model_path):
        print(f"Error: no model - {model_path}")
        return

    print(f"load model: {model_path}")
    model = YOLO(model_path)


    print(f"\nstart to detect: {image_path}")
    results = model.predict(
        source=image_path,
        save=True,          
        show=False,         
        verbose=True       
    )

    
