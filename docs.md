# Tech stack 

Environments:
- Windows server 2022 
(My Debian 14 brokes, currrent under repair so I have to use another old server)
- python 3.0.13 from Visual Studio 2022
- Using Ultralytics maintained Yolo Repository
Tasks:
- Target: correctly detect "Armored Core"
- Target: correctly detect modern tanks and drones
- Target: Execute some simple task like recognize coco80's objects


# Memos, Tech knowledge maps, commands, etc.
 all dir refered inside this docs is OS-specific, which means:----only works on my Windwos Server 2022
on my server, install location is: C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\Lib\site-packages\ultralytics

 all `yolo` inside this docs refers to the yolo cli tools or yolo model, based on context
## concepts

### maintainers: 
- Joseph Redmon, author [repo addr](https://github.com/pjreddie/darknet)
- Alexey Bochkovskiy, maintainer [repo addr](https://github.com/AlexeyAB/darknet)
- YOLOX, actually MeiTuan foor delivery, no repo addr since We don't need to deploy these models.
- Ultralytics, maintainers [repo addr](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
what we will use: 
- UltraLytics maintained YOLO repo, it has cmd tools, easy for quick preview.

   `yolo` : command line tools, quickly calls the pretrained model, simply download datasets, run testings
   [cli docs](https://docs.ultralytics.com/usage/cli/)

## yolo model infos

### configs Specs
yolo pretrained model can detected all object in coco80 class: ultralytics/cfg/datasets/coco.yaml

yolo implicitly use config under dir: C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\Lib\site-packages\ultralytics\cfg

### Datasets Specs
Datasets: pure image with some annotation. annotation often exists as txt files, indicating the box bound of detect target. it also has multiple format.
(I was expecting researchers create a exclusive format for annotating dates...)

yolov8 has many sub-models. here are their infos: [Model spec sheets](https://docs.ultralytics.com/zh/models/yolov8/#performance-metrics:~:text=%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87-,%E6%80%A7%E8%83%BD,-%E6%A3%80%E6%B5%8B%20(COCO))

yolov8 has its own exclusive format of datesets, named `yolov8`, structure below:
```
tank_dataset/
├─ images/
│  ├─ train/         # jpg/png
│  └─ val/
└─ labels/
   ├─ train/         # 同名 txt
   └─ val/
```

or: 
```
drone_dataset/
├── test
│   ├── images
│   └── labels #txt notation files
|---labels.cache #if some dataset-relative error occurs, delete this file!
├── train
│   ├── images
│   └── labels
|---labels.cache #if some dataset-relative error occurs, delete this file!
└── valid
    ├── images
    └── labels
|---labels.cache #if some dataset-relative error occurs, delete this file!
```

Warning: yolo will cache the label detection result under train or val datasets. if you got any dataset error, first thing to do is delete all label cache!!!!! this error controls me for 1 hour

### Codes

#### datasets 
Preprocess: if preprocess on dataset before downloading is possible always do it. you need:
- Orentation: all photos point to same direction, up, e.g. This reduce your machine pressure when training
- resize: Always resize if possible. resize in `yolov8n` can be square 640 or 416
- High constract: if your dataset is small consider this
- Filter Null: use this is recommended. Otherwise you have to config yolo filtered null training example. Training with less Null example makes `yolo` focus on training class

Augementation: If your dataset is small, it can add some blur, rotation or noise on pictures. However yolo has built-in augmentation machanisims. So this option is not necessary

Available model params on dataset augementation is [here](https://docs.ultralytics.com/zh/modes/train/#train-settings:~:text=%E8%B0%83%E6%95%B4%E6%89%B9%E9%87%8F%E5%A4%A7%E5%B0%8F%E3%80%82-,%E5%A2%9E%E5%BC%BA%E8%AE%BE%E7%BD%AE%E5%92%8C%E8%B6%85%E5%8F%82%E6%95%B0,-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF)


#### checkpoints params

yolo checkpoint mechanisms:
```
from ultralytics import YOLO

model = YOLO("path/to/last.pt")  # load a partially trained model

results = model.train(resume=True)
```
通过设置 resume=True， train 函数将从上次停止的地方继续训练，使用存储在 'path\/to\/last.pt' 文件中的状态。如果省略 resume 参数或将其设置为 False， train ，该函数将启动新的训练会话。

请记住，默认情况下，检查点会在每个 epoch 结束时保存，或者使用 save_period 参数按固定间隔保存，因此您必须至少完成 1 个 epoch 才能恢复训练运行。

YOLO 会在 runs/train/exp/ 目录（或者 runs/train/exp2/, exp3/ … 依次递增）下保存结果。

里面通常有：

```
weights/
    last.pt   # 上一个 epoch 结束时的模型权重 + 状态 Keep training use this one
    best.pt   # 根据验证集指标选出来的最佳模型 terminated this training and put model into warfare use this one
results.csv  # 训练日志
events.out.tfevents.*  # TensorBoard 日志

```

检查点信息是存储在 `last.pt` 文件里，所以只要有这个文件，就能 resume=True 继续训练。

all yolo operations comes with default locations. so try to follow the file struction conventions

#### models params
Import model params:
```
optimizer: always use AdamW
save_period: don't use 1 if you want to save disk space
class: let model keep focusing on some class
warm-Up
lr-relative
cos_lr
profile
dropout: prevent overfitting
mask: segment input, instead of full picture input
```

Export model method params:
```
device : since yolo needs to infered once when output model, this params indicated what device is used to infered.
pt just a file with bunch of params and numbers, it is cross-platform. not cpu/gpu optimized.
int8: if enable, loss some precision for better performance, looks fits us
dataset: correspond to above in8 params. 
```

增强设置和超参数
if your dataset is good you don't even need these params



### CMD emos

`yolo predict model=xxxxx source=xxxxx`
Noticed it directly download the model to execution dir, need to figure how to specific model location.

results default generate same image with box boundarys in `${run_directory}/runs/detect/predict${number}/xxx.jpg`


`yolo predict model=xxxxx source=xxxxx` quick predict using pretrained model



# used examples:
all used images example can be found under `./images`
all detected result can be found under `./runs/detect/*`
here are some links to original pictures
[drone1](https://cdn.defenseone.com/media/img/cd/2024/10/11/8273054/860x394.jpg)
[drone2](https://cdn.thewirecutter.com/wp-content/media/2023/11/dronesforphotovideo-2048px-DSC4837-3x2-1.jpg?auto=webp&quality=75&crop=3:2&width=1024)
[drone3](https://www.zuken.com/us/wp-content/uploads/sites/12/2023/03/dji-dji-agriculture-agras.png)


# P.S.
kaggle数据集真不行，真要找逆天数据集还得是roboflow牛逼
hard to find noted dataset of mechs, robots, armored cores.......
need stable and rich data sources
-i https://mirrors.ustc.edu.cn/pypi/simple
Most of the time spend on training and reading code-relative docs......

drones sample approximate: 41639
tanks sample approximate: 2134

