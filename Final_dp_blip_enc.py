import cv2
#import PIL
#from split_image import split_image
#import numpy as np
import torch
from transformers import AutoProcessor, BlipModel
from PIL import Image
#import requests
from sklearn import preprocessing


def get_dp_blip_enc(uid, imgid):

    readfile = FOLDER + '\\' + uid + '\\' + uid + '_profile' + '.jpg'  
#    readfile = filepath
    
#    image_len = 1024
    
    
    # Reading the image using imread() function
#    try:
    rgb_image = cv2.imread(readfile)
    if rgb_image is None:
#    if(rgb_image.any() == None):
        print("Got no image here", uid, imgid)
        enc = torch.zeros(4608, dtype = torch.float32)
        return enc
    rgb_image = cv2.resize(rgb_image, (image_len,image_len))    
    
    M = rgb_image.shape[0]//3
    N = rgb_image.shape[1]//3
    
    tiles = [rgb_image[x:x+M,y:y+N] for x in range(0,rgb_image.shape[0],M) for y in range(0,rgb_image.shape[1],N)]
    
# =============================================================================
#     print(len(tiles)) 
#     cv2.imshow('tile 1',tiles[0])
#     cv2.imshow('tile 2',tiles[1])
#     cv2.imshow('tile 3',tiles[2])
#     cv2.imshow('tile 4',tiles[4])
#     cv2.imshow('tile 5',tiles[5])
#     cv2.imshow('tile 6',tiles[6])
#     cv2.imshow('tile 7',tiles[6])
#     cv2.imshow('tile 8',tiles[7])
#     cv2.imshow('tile 9',tiles[8])
#     cv2.waitKey(0)
# =============================================================================
    
#    print(tiles[0])
#    pil_tile = Image.fromarray(tiles[0].astype('uint8'), 'RGB')
#    pil_tile = Image.fromarray(np.uint8(tiles[0]))
    
    pil_tile = cv2.cvtColor(tiles[0], cv2.COLOR_BGR2RGB) 
    pil_tile = Image.fromarray((pil_tile)).convert('RGB')
#    pil_tile.show()
    pil_tiles = []
    for tile in tiles:
        temp = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB) 
        temp = Image.fromarray((temp)).convert('RGB')
        pil_tiles.append(temp)
    
    
#    pil_tiles[0].show("0")
#    pil_tiles[1].show("1")
        
#    print(pil_tiles[0])
#    inputs = processor(images=pil_tiles[0], return_tensors="pt")
#    image_features = model.get_image_features(**inputs)
#    print(image_features.shape)
#    print(image_features.dtype)
#    print(len(pil_tiles))
#    print(len(tiles))
    
    
    
        
    tile_tensor = torch.empty((0, 512), dtype=torch.float32)    
    
    for tile in pil_tiles:
        
#        print(len(tile))
#        tile.show("0")
        
        curr_tile = processor(images=tile, return_tensors="pt")
        curr_features = model.get_image_features(**curr_tile)
#        print(curr_features.shape)
        tile_tensor = torch.vstack((tile_tensor,curr_features))
#        break

        
#    print(tile_tensor.shape)
#    print(tile_tensor.dtype)
    
#    flat_img_enc = torch.flatten(tile_tensor)
#    flat_img_enc = flat_img_enc.detach()
    flat_img_enc = torch.flatten(tile_tensor)
    flat_img_enc = flat_img_enc.detach().cpu().numpy()
#    encoded_doc = encoded_doc.flatten() 
    img_enc = flat_img_enc.reshape(1, -1)
#    print(type(img_enc))
    normalized_enc_np = preprocessing.normalize(img_enc)
#    print(type(normalized_enc_np))
    
    normalized_enc = torch.from_numpy(normalized_enc_np)
#    print(type(normalized_enc))
#    print(normalized_enc.dtype)
    
    normalized_enc = normalized_enc.to(torch.float32)
#    print(normalized_enc.dtype)
    
    normalized_enc_flat = normalized_enc.flatten()        
        
#    print(flat_img_enc.shape)
#    print(flat_img_enc.dtype)    
    
    return normalized_enc_flat

image_len = 510
    
FOLDER = r'F:\PhD\Datasets\twitter-collection-master\twitter-collection-master\Final Dataset v11'

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
if __name__ == "__main__":
#    print ("Executed when invoked directly")

    
    uid = "222188599"
    imgid = '48'
    
    enc = get_dp_blip_enc(uid,imgid)
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(enc.shape)
    print(enc)
    print(torch.max(enc))
    print(torch.min(enc))
#    print(np.max(enc))
#    print(np.min(enc))
#    print(enc[2120])
#    print(enc[100])
    
    
    
    
    
    
    
    
    
    
    
    











