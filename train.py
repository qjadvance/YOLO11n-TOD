import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"    
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n-TOD.yaml') # YOLO11
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'E:\BaiduNetdiskDownload\yolo11n-SOD\yolo11n-SOD\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=3,
                batch=8,
                close_mosaic=0, 
                workers=4,
                # device='0,1', 
                optimizer='SGD',
                # patience=0,
                # resume=True,
                amp=False, 
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
