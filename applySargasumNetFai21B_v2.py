"""Test for SegNet 



python3 applyStrideLucModelResidual12B2CL.py --input /data/Sargassum/SargassumHP/Sargassum/Images12B/S2B_20180929_12b.tif --output result20180929_12b_stride_PlusBGAug_ALL.tif  --weights segnet_Sargassum_valid_best.pth --cuda


python ~/soft/Sargassum2D_2021/applyStrideLucModelResidual12B2CL.py --weights ~/soft/Sargassum2D_2021/Sargassum64_12B_NEW_75000_V2_valid_best_best_v2.pth --cuda --input S2B_MSIL2A_20200129T143659_N0213_R096_T20PRV_20200129T165410.tif --output S2B_MSIL2A_20200129T143659_N0213_R096_T20PRV_20200129T165410_SAR.tif  

python3 applyModelResidual12B2CL.py --input /data/Sargassum/SargassumHP/Sargassum/Sargassum32_12B/Images/20180919_125.tif --output /tmp/result20180919_125.tif  --weights segnet_Sargassum_valid_best12B.pth --cuda
  
  
python3 predictImageSargasumModels3D12B.py  --img /data/Sargassum/ImagesCrop100x100/S2B_20180919_crop_800x1150_100x100_12b.tif  --output S2B_20180919_crop_800x1150_100x100_predict.png --weights  weights/model_lc_best5000_3D_12B.pth
python3 predictImageSargasumModels3D12B.py  --img S2B_20180919_crop_800x1150_100x100_12b.tif  --output S2B_20180919_crop_800x1150_100x100_predict.png --weights  weights/model_lc_best5000_3D_12B.pth
 
python3 predictImageSargasumModels3D12B.py  --img ../Sargassum/Images12B/S2B_20180919_12b.tif --output S2B_20180919_12bresult.tif  --weights  weights/model_lc_best5000_3D_12B.pth --cuda

python3 predictImageSargasumModels5x5_3D12B.py  --img ../Sargassum/Images12B/S2B_20180929_12b.tif --output S2B_20180929_5x5_12bresult.tif  --weights  weights/model_lc_best_5x5_3D_12B.pth --cuda


"""

from __future__ import print_function
 
import numpy as np
import torch
import argparse
import os
from osgeo import gdal
 
from sargassum_model_v2 import  SargassumNet2DV3L_3x3_Residual as SargassumNet 

from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import sys
import cv2
#
#def load_image(path=None):
#        print(path)
#        raw_image = Image.open(path)
#        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
#        imx_t = np.array(raw_image, dtype=np.float32)/255.0
#           
#        return imx_t
# 
  
parser = argparse.ArgumentParser(description='Predict Image Sargassum')
 
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--weights', default=os.path.join(os.path.split(sys.argv[0])[0],"Sargassum21b_128_FAI_V15_Aug4.pth"))
parser.add_argument('--channels', type=int, default=21)
parser.add_argument('--cuda', action='store_true', help='use cuda',default=False)
parser.add_argument('--size', default=128, type=int,help='size of stride')
parser.add_argument('--model', default="64", type=str,help='model Sargaum  32 |64')
parser.add_argument('--dims', type=str, default="(32,64)" )
parser.add_argument('--no_nor', action='store_true', help='use normalize')
parser.add_argument('--thresold', type=int, default=3 ) 
parser.add_argument('--save_nb', action='store_true', help='use normalize')
                  
args = parser.parse_args()

print("option",args.cuda)
device="cpu"
 
if args.cuda   :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("DEVICE",device)
 
# RGB input
input_channels = args.channels
#  
output_channels = 1

print("Create modele")
 
dim=eval(args.dims)
model = SargassumNet(input_channels,output_channels,dims=dim) 
 
if  args.cuda :
    model=model.cuda()
else:
    model=model.cpu()

print('Model nb parameters:', sum(param.numel() for param in model.parameters()))

if args.weights != "None" :
    
    file=args.weights
    print("load",file)
    net_weights=torch.load(file, map_location=device)
    model.load_state_dict(net_weights)          
    model.eval()
    
gpu=False
   
size=args.size
 
mean=\
[ 0.09729894,  0.08877715,  0.08311013,  0.07391645,  0.09017547,  0.10118437,
  0.10725799,  0.10664269,  0.10172991,  0.17433181,  0.06167458,  0.04890004]
std=\
[ 0.00434938,  0.00643737,  0.00729247,  0.00673562,  0.00925067 , 0.00968437,
  0.01022551,  0.0092595,   0.01164183, 0.00630978,  0.00411137,  0.0039594 ]
 
mean=[0.41241804, 0.43961947 ,0.41587106, 0.30029273, 0.26220613, 0.12625736,\
      0.10230557 ,0.07557606 ,0.06518111, 0.07567135 ,0.07038689 ,0.04778447, \
      0.02257335 ,0.02585678, 0.05598966, 0.05317822, 0.03232022, 0.02597062, \
       0.02325512, 0.02224471 ,0.02122725]
std=[0.00087046, 0.00085735, 0.00081165, 0.00071333, 0.00075845, 0.0004673,\
     0.00046252, 0.00034428 ,0.00033373 ,0.00038784, 0.00038276 ,0.00036716,\
      0.00037718, 0.00034352, 0.00068981, 0.00034353, 0.0002783,  0.00044168,\
      0.00035782, 0.00040527, 0.00126199]

def load_image(  path=None,channels=21,normalize=False):        
        #print(path)        
        ds_img = gdal.Open(path )
        
        #print("NB bands ",ds_img.RasterCount)
        cols = ds_img.RasterXSize
        rows = ds_img.RasterYSize
        bands = ds_img.RasterCount
        
        print("Image size b" ,bands,"Xsize",rows,"Ysize",cols )
        if ds_img.RasterCount !=  channels :
            print("ERROR   BAND  NUMBER  ",channels,ds_img.RasterCount)
            quit()
        
        imx_t=np.empty([channels, rows, cols], dtype=np.float32)
        
        for b in range(1,ds_img.RasterCount +1) :
            channel = np.array(ds_img.GetRasterBand(b).ReadAsArray())
            
#            print(channel.shape)    
            # channel = cv2.resize(channel, ( size,  size))
#            
#            print(b , "shape" ,channel.shape)
            #print( "b",b," min", np.amin(channel),"max",np.amax(channel))
            np_float=np.array(channel, dtype=np.float32)/12500.0
            np_float[np_float > 1] =1
            imx_t[b-1,:,:]=np_float
             #normalise 
         
        if  normalize :   
            print("NOR",channels)
            for b in range(channels) :
              imx_t[b,::]=(imx_t[b,::]  -mean [b] )/std [b] 
     
        return imx_t
  
imgnp=load_image(args.input,channels=args.channels,normalize=not args.no_nor)

print("SIZE",imgnp.shape)

palettedata = [0,0,0,255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255, 127,0,0, 0,127,0,  237, 127, 16  , 127,127,0, 127,0,127, 0,127,127]  

print("img size ",imgnp .shape)
(height,width)=imgnp.shape[1:3]
print("img size ", height,width)

resimage = np.zeros( ( height,width ),dtype=np.uint8)
#resimage.putpalette(palettedata )

imgs = torch.autograd.Variable(torch.from_numpy(imgnp)) 

k_size = args.size 

stride = k_size//2

bord=16
if args.size <= 32:
    stride =16
    bord =8

if args.size ==  64 :
    stride = 32
    bord =16
    
if args.size ==  64 :
    stride = 44
    bord =10 
        
if args.size ==  64 :
    stride = 48
    bord =8   
    
if args.size ==  128 :
    stride = 92
    bord =6

if args.size ==  256 :
    stride = 128
    bord =8       
#if args.size ==  64 :
#    stride = 48
#    bord =8
    
print("Kernel",k_size,"stride",stride,"bordure",bord)

#
#if args.size  > 32:
#    stride = int(k_size/1.5)
    
#if args.size  >32:
#    stride = args.size -16 

#print(args.size,stride,imgs.size())

nb=((height-stride+2)//stride) *((width-stride+2)//stride)

i=0
for _y in range (0,height-stride+1,stride): 
    y=_y
    for x  in range (0,width-stride+1,stride):
                 
        print("\r",i,'/',nb,end=" ")
        #print(y,y+k_size ,height,"X",x,x+k_size,width)
        if y+k_size >= height :
            y = height -k_size
            
        if x+k_size  >= width :
            x=width  - k_size        
    
        #print(y,(y+k_size),x,(x+k_size))
        crop_img = imgs[0:,y:y+k_size,x:x+k_size]
        
        
        #print(crop_img.shape)
#            zero=0
#            for yy in range(height ):
#                        for xx in range(width ):
#                              if imgcv[im][0,yy,xx]== 0:
#                          
#                                   
#                                    zero+=1
        
        
        crop_img =crop_img.unsqueeze(0).to(device)
    
    #    print(p,img.size())
        output = model(crop_img)
        
        
        output= output.squeeze()
        output= output.squeeze()
#                output=torch.squeeze(output,1)
        pred = output.detach().cpu().numpy()
        
        
        pred*=255 
                
        pred=np.clip(pred,0,255)
         
        pred=pred.astype(np.uint8)
        if args.thresold != 0 :
            pred[pred<= args.thresold]=0
 
    
#        pred=pred.transpose((1,0))
     
        
#        imgs[0:,y:y+k_size,x:x+k_size]
#        
#        resimage[y+bord -1 :y+k_size-bord,x+bord-1:x+k_size-bord]=0
#        pred[bord -1 :k_size-bord, bord-1:k_size-bord]=0
        
        
        resimage[y+bord -1 :y+k_size-bord,x+bord-1:x+k_size-bord]=pred[bord -1 :k_size-bord, bord-1:k_size-bord]
        
#        print((y+bord -1 ,y+k_size-bord),(x+bord-1,x+k_size-bord))           
             
        i+=1  
        
      
cv2.imwrite(args.output,resimage ) 
print('\nsave to ', args.output  ) 

if args.save_nb == True :
    resimage[resimage>0]=255
    
    nom2=args.output[:-4]+"_NB.png"
    cv2.imwrite(nom2,resimage )    
  
sys.exit()  


 


 