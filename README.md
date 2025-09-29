# Project Structure 

`/dataset_sources` including raw dataset (due to big size this dir is in gitignore list)

`datasets` including training dataset (due to big size this dir is in gitignore list)

`models` includig model weight files. load them to use model. Always use `best.pt`

`runs` including every predict result

`images` including test images

# How to check my result
- read `./docs.md`
if you don't have markdown reader just google "convert markdown to pdf" and upload the docs.md to get a converted pdf

- use command line to run `yolo predict model='./models/drones/best.pt' source='location/to/your/file`
- or load the python file in a interactive mode, run `python i`, then inside python run `import trainmodel as ac`, `ac.yolo_predict('./models/drones/best.pt', 'url_or_location_to_your_image_file', False)`
the third params decide if using proxy or not when the url points to a online picture. If you want to use proxy please change the addr and port accordingly, otherwise set it as False

NOTICED: if you want to detect tanks, use the model under models/tanks directory. If you want to detect drones, use the model under models/drones directory.

They are trained with different params and method, and for efficient and performance I modify the original model. That means these two models can NO LONGER detect the original 80 class, but ONLY tanks or drones. This is for maximum the performance.

the `source` parameter support: 
1. location to image file
2. network url to image file (sometimes it is buggy, so please use 1 if it is possible)
