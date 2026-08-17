import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
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
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练F
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )