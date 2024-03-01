import numpy as np
import cv2
import imutils
import pywt
import torch
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from sklearn import preprocessing


def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    temp = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    return cv2.resize(temp, (image_len,image_len))
#    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def get_dp_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1):
#        All are active by default
#        put wavelet = 0 if wavelet is not required, 
#            rgb = 0 if rgb is not required,
#            way1 = 0 if way1 of making collage is not required
#            way2 = 0 if way2 of making collage is not required
            
#    749003_profile.jpg    
    readfile = FOLDER + '\\' + uid + '\\' + uid + '_profile' + '.jpg'  
#    readfile = filepath
    
    image_len = 512
    wavelet_len = 256
    
    
    # Reading the image using imread() function
#    try:
    rgb_image = cv2.imread(readfile)
    if rgb_image is None:
#    if(rgb_image.any() == None):
        print("Got no dp here", uid, imgid)
        enc = torch.zeros(4192, dtype = torch.float32)
        return enc
    rgb_image = cv2.resize(rgb_image, (image_len,image_len))
    grey_image = cv2.imread(readfile,0)
    grey_image = cv2.resize(grey_image, (image_len,image_len))
#    except:
#        enc = torch.zeros(4192, dtype = torch.float32)
#        return enc
    
    try:
        # Using cv2.split() to split channels of coloured image 
        b,g,r = cv2.split(rgb_image)
        
    # =============================================================================
    # #    Displaying the original BGR image
    #     cv2.imshow('Original_Image', rgb_image)
    # =============================================================================
    
        grey_0 = grey_image
        blue_0 = b
        green_0 = g
        red_0 = r
        
        
        # 22.5 degree rotation of all
        rotated_grey_22 = imutils.rotate_bound(grey_image, -22.5)
        rotated_blue_22 = imutils.rotate_bound(blue_0, -22.5)
        rotated_green_22 = imutils.rotate_bound(green_0, -22.5)
        rotated_red_22 = imutils.rotate_bound(red_0, -22.5)
        
    # =============================================================================
    #     cv2.imshow("Rotated 22 Grey Image", rotated_grey_22)
    #     cv2.imshow("Rotated 22 Blue Image", rotated_blue_22)
    #     cv2.imshow("Rotated 22 Green Image", rotated_green_22)
    #     cv2.imshow("Rotated 22 Red Image", rotated_red_22)
    # =============================================================================
        
        # 45 degree rotation of all
        rotated_grey_45 = imutils.rotate_bound(grey_image, -45)
        rotated_blue_45 = imutils.rotate_bound(blue_0, -45)
        rotated_green_45 = imutils.rotate_bound(green_0, -45)
        rotated_red_45 = imutils.rotate_bound(red_0, -45)
        
        # =============================================================================
        # cv2.imshow("Rotated 45 Grey Image", rotated_grey_45)
        # cv2.imshow("Rotated 45 Blue Image", rotated_blue_45)
        # cv2.imshow("Rotated 45 Green Image", rotated_green_45)
        # cv2.imshow("Rotated 45 Red Image", rotated_red_45)
        # =============================================================================
        
        # 77.5 degree rotation of all
        rotated_grey_77 = imutils.rotate_bound(grey_image, -77.5)
        rotated_blue_77 = imutils.rotate_bound(blue_0, -77.5)
        rotated_green_77 = imutils.rotate_bound(green_0, -77.5)
        rotated_red_77 = imutils.rotate_bound(red_0, -77.5)
        
        # =============================================================================
        # cv2.imshow("Rotated 77 Grey Image", rotated_grey_77)
        # cv2.imshow("Rotated 77 Blue Image", rotated_blue_77)
        # cv2.imshow("Rotated 77 Green Image", rotated_green_77)
        # cv2.imshow("Rotated 77 Red Image", rotated_red_77)
        # =============================================================================
        
        
        # Add all to list of rotated images
        img_rotation_collection = []
        
        img_rotation_collection.append(grey_0)
        img_rotation_collection.append(blue_0)
        img_rotation_collection.append(green_0)
        img_rotation_collection.append(red_0)
        
        img_rotation_collection.append(rotated_grey_22)
        img_rotation_collection.append(rotated_blue_22)
        img_rotation_collection.append(rotated_green_22)
        img_rotation_collection.append(rotated_red_22)
        
        img_rotation_collection.append(rotated_grey_45)
        img_rotation_collection.append(rotated_blue_45)
        img_rotation_collection.append(rotated_green_45)
        img_rotation_collection.append(rotated_red_45)
        
        img_rotation_collection.append(rotated_grey_77)
        img_rotation_collection.append(rotated_blue_77)
        img_rotation_collection.append(rotated_green_77)
        img_rotation_collection.append(rotated_red_77)
        
            
    # =============================================================================
    #     # Check the collection of rotated images
    #     print(len(img_rotation_collection))
    #     for i in range(16):
    #         cv2.imshow(str(i),img_rotation_collection[i])
    # =============================================================================
        
        
        
        # Wavelet Transform the Images
        
        # =============================================================================
        # LL1, (LH1, HL1, HH1) = pywt.dwt2(img_rotation_collection[4], 'bior1.3')
        # cv2.imshow("Trial image", img_rotation_collection[4])
        # #cv2.imshow("Trial image ll", LL1)
        # cv2.imshow("Trial image lh", LH1)
        # cv2.imshow("Trial image hl", HL1)
        # #cv2.imshow("Trial image hh", HH1)
        # =============================================================================
        
        # Wavelet image list
        img_wavelet_lh = []
        img_wavelet_hl = []
        
        
        for i in range(16):
            LL, (LH, HL, HH) = pywt.dwt2(img_rotation_collection[i], 'bior1.3')
            img_wavelet_lh.append(LH)
            img_wavelet_hl.append(HL)
        
        img_wavelet_collection = []
        
        for image in img_wavelet_lh:
            img_wavelet_collection.append(image)
            
        for image in img_wavelet_hl:
            img_wavelet_collection.append(image)
        
        #print(len(img_wavelet_collection))
        
        
        
        #   store 0 degree
        for i in range(0,4):
        #    cv2.imshow(str(i),img_wavelet_lh[i])
            img_wavelet_lh[i] = cv2.resize(img_wavelet_lh[i], (wavelet_len,wavelet_len))
            
        
        #  22 degree rotate back
        for i in range(4,8):
        #    cv2.imshow(str(i),img_wavelet_lh[i])
            rerotated = imutils.rotate_bound(img_wavelet_lh[i], 22.5)
    #        cv2.imshow("re Rotated Image", rerotated)
            
            crop_rerotated = crop(rerotated)
    #        cv2.imshow("crop re Rotated Image", crop_rerotated) 
               
            img_wavelet_lh[i] = crop_rerotated
            img_wavelet_lh[i] = cv2.resize(img_wavelet_lh[i], (wavelet_len,wavelet_len))
            
    #        cv2.imshow(str(i), img_wavelet_lh[i]) 
        #    break
        
        #  45 degree rotate back
        for i in range(8,12):
        #    cv2.imshow(str(i*2),img_wavelet_lh[i])
            rerotated = imutils.rotate_bound(img_wavelet_lh[i], 45)
        #    cv2.imshow("re Rotated Image", rerotated_grey_22)
            
            crop_rerotated = crop(rerotated)
        #    cv2.imshow("crop re Rotated Image", crop_rerotated_grey_22) 
               
            img_wavelet_lh[i] = crop_rerotated
            img_wavelet_lh[i] = cv2.resize(img_wavelet_lh[i], (wavelet_len,wavelet_len))
            
    #        cv2.imshow(str(i), img_wavelet_lh[i]) 
        #    break
        
        #  77 degree rotate back
        for i in range(12,16):
        #    cv2.imshow(str(i*2),img_wavelet_lh[i])
            rerotated = imutils.rotate_bound(img_wavelet_lh[i], 77.5)
        #    cv2.imshow("re Rotated Image", rerotated_grey_22)
            
            crop_rerotated = crop(rerotated)
        #    cv2.imshow("crop re Rotated Image", crop_rerotated_grey_22) 
               
            img_wavelet_lh[i] = crop_rerotated
            img_wavelet_lh[i] = cv2.resize(img_wavelet_lh[i], (wavelet_len,wavelet_len))
            
    #        cv2.imshow(str(i), img_wavelet_lh[i]) 
        #    break
        
        
        #  0 degree rotate back
        for i in range(0,4):
        #    cv2.imshow(str(i),img_wavelet_hl[i])
            img_wavelet_hl[i] = cv2.resize(img_wavelet_hl[i], (wavelet_len,wavelet_len))
        
        #  22 degree rotate back
        for i in range(4,8):
        #    cv2.imshow(str(i*2),img_wavelet_hl[i])
            rerotated = imutils.rotate_bound(img_wavelet_hl[i], 22.5)
        #    cv2.imshow("re Rotated Image", rerotated_grey_22)
            
            crop_rerotated = crop(rerotated)
        #    cv2.imshow("crop re Rotated Image", crop_rerotated_grey_22) 
               
            img_wavelet_hl[i] = crop_rerotated
            img_wavelet_hl[i] = cv2.resize(img_wavelet_hl[i], (wavelet_len,wavelet_len))
            
        #    cv2.imshow(str(i), img_wavelet_hl[i]) 
        #    break
        
        #  45 degree rotate back
        for i in range(8,12):
        #    cv2.imshow(str(i*2),img_wavelet_hl[i])
            rerotated = imutils.rotate_bound(img_wavelet_hl[i], 45)
        #    cv2.imshow("re Rotated Image", rerotated_grey_22)
            
            crop_rerotated = crop(rerotated)
        #    cv2.imshow("crop re Rotated Image", crop_rerotated_grey_22) 
               
            img_wavelet_hl[i] = crop_rerotated
            img_wavelet_hl[i] = cv2.resize(img_wavelet_hl[i], (wavelet_len,wavelet_len))
            
        #    cv2.imshow(str(i), img_wavelet_hl[i]) 
        #    break
        
        #  77 degree rotate back
        for i in range(12,16):
        #    cv2.imshow(str(i*2),img_wavelet_hl[i])
            rerotated = imutils.rotate_bound(img_wavelet_hl[i], 77.5)
        #    cv2.imshow("re Rotated Image", rerotated_grey_22)
            
            crop_rerotated = crop(rerotated)
        #    cv2.imshow("crop re Rotated Image", crop_rerotated_grey_22) 
               
            img_wavelet_hl[i] = crop_rerotated
            img_wavelet_hl[i] = cv2.resize(img_wavelet_hl[i], (wavelet_len,wavelet_len))
            
        #    cv2.imshow(str(i), img_wavelet_hl[i]) 
        #    break
        
        # =============================================================================
        # for i in range(16):
        # #    cv2.imshow(str(i), img_wavelet_lh[i])
        #     print(img_wavelet_lh[i].shape)
        # #    break
        #     
        # for i in range(16):
        # #    cv2.imshow(str(i+1), img_wavelet_hl[i])
        #     print(img_wavelet_hl[i].shape)
        # #    break
        # =============================================================================
        
        
        #    Collage the Image in 2 ways
        #    Way 1
            
        Horizontal1=np.hstack([img_wavelet_lh[0],img_wavelet_lh[1],img_wavelet_lh[8],img_wavelet_lh[9],img_wavelet_hl[0],img_wavelet_hl[1],img_wavelet_hl[8],img_wavelet_hl[9]])
        Horizontal2=np.hstack([img_wavelet_lh[2],img_wavelet_lh[3],img_wavelet_lh[10],img_wavelet_lh[11],img_wavelet_hl[2],img_wavelet_hl[3],img_wavelet_hl[10],img_wavelet_hl[11]])
        Horizontal3=np.hstack([img_wavelet_lh[4],img_wavelet_lh[5],img_wavelet_lh[12],img_wavelet_lh[13],img_wavelet_hl[4],img_wavelet_hl[5],img_wavelet_hl[12],img_wavelet_hl[13]])
        Horizontal4=np.hstack([img_wavelet_lh[6],img_wavelet_lh[7],img_wavelet_lh[14],img_wavelet_lh[15],img_wavelet_hl[6],img_wavelet_hl[7],img_wavelet_hl[14],img_wavelet_hl[15]])
        
        collage1 =np.vstack([Horizontal1,Horizontal2,Horizontal3,Horizontal4])
    #    collage1 *= (255.0/collage1.max())
    #    norm_collage1 = (255*(collage1 - np.min(collage1))/np.ptp(collage1)).astype(int)
        
    #    collage1_resize = cv2.resize(collage1, (1500,750))
    #    cv2.imshow("Final Collage 1",collage1_resize)
    #    print(np.min(norm_collage1),np.max(collage1))
    #    cv2.imwrite("Final Collage 1.jpg", collage1)
        
        
        #    Way 2
        
        Horizontal10=np.hstack([img_wavelet_lh[0],img_wavelet_lh[4],img_wavelet_lh[1],img_wavelet_lh[5],img_wavelet_lh[2],img_wavelet_lh[6],img_wavelet_lh[3],img_wavelet_lh[7]])
        Horizontal20=np.hstack([img_wavelet_lh[8],img_wavelet_lh[12],img_wavelet_lh[9],img_wavelet_lh[13],img_wavelet_lh[10],img_wavelet_lh[14],img_wavelet_lh[11],img_wavelet_lh[15]])
        Horizontal30=np.hstack([img_wavelet_hl[0],img_wavelet_hl[4],img_wavelet_hl[1],img_wavelet_hl[5],img_wavelet_hl[2],img_wavelet_hl[6],img_wavelet_hl[3],img_wavelet_hl[7]])
        Horizontal40=np.hstack([img_wavelet_hl[8],img_wavelet_hl[12],img_wavelet_hl[9],img_wavelet_hl[13],img_wavelet_hl[10],img_wavelet_hl[14],img_wavelet_hl[11],img_wavelet_hl[15]])
        
        collage2 =np.vstack([Horizontal10,Horizontal20,Horizontal30,Horizontal40])
        
    #    collage2_resize = cv2.resize(collage2, (1500,750))
    #    cv2.imshow("Final Collage 2",collage2_resize)
    #    cv2.imwrite("Final Collage 2.jpg", collage2,cmap="gray")
        
        #    Encode the collages
        
        v1 = ViT(
            image_size = 2048,
            patch_size = 256,
            num_classes = 1000,
            dim = 32,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            channels = 1
        )
        
        v1 = Extractor(v1)
        
        collage1_t = torch.from_numpy(collage1)
        collage2_t = torch.from_numpy(collage2)
        
        collage1_t = collage1_t[None, None,:, :]
        collage2_t = collage2_t[None, None,:, :]
        
        collage1_t = collage1_t.to(torch.float32)
        collage2_t = collage1_t.to(torch.float32)
        
        logits1, embeddings1 = v1(collage1_t)
        
        logits2, embeddings2 = v1(collage2_t)
        
        embeddings1 = embeddings1.flatten()
        embeddings2 = embeddings2.flatten()
        
    #    print(embeddings1.shape)
    #    print(embeddings2.shape)
        
        #    Encode the image
        
        v2 = ViT(
            image_size = 512,
            patch_size = 64,
            num_classes = 1000,
            dim = 32,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            channels = 3
        )
        
        v2 = Extractor(v2)
        
        rgb_image_t = torch.from_numpy(rgb_image)
        rgb_image_t = torch.permute(rgb_image_t, (2, 0, 1))
        rgb_image_t = rgb_image_t[None,:, :]
        rgb_image_t = rgb_image_t.to(torch.float32)
        
        #print(rgb_image_t.shape)
        
        logits1, embeddings3 = v2(rgb_image_t)
        
        embeddings3 = embeddings3.flatten()
    #    print(embeddings3.shape)
        
    #    wavelet = 1, rgb = 1, way1 = 1, way2 = 1
        if (wavelet == 0):
            embeddings1 = torch.zeros(1056, dtype = torch.float32)
            embeddings2 = torch.zeros(1056, dtype = torch.float32)
         
        if (rgb == 0):
            embeddings3 = torch.zeros(2080, dtype = torch.float32)
            
        if (way1 == 0):
            embeddings1 = torch.zeros(1056, dtype = torch.float32)
        
        if (way2 == 0):
            embeddings2 = torch.zeros(1056, dtype = torch.float32)
        
            
        img_encoding = np.hstack([embeddings1,embeddings2,embeddings3])
    
    except:
        print("error with borders", uid, imgid)
        enc = torch.zeros(4192, dtype = torch.float32)
        return enc
    
    
    img_encoding = img_encoding.reshape(1, -1)
    normalized_enc = preprocessing.normalize(img_encoding)
    normalized_enc_flat = normalized_enc.flatten()
#    print(img_encoding.shape)
    
    return normalized_enc_flat

readfile = "96573740_36 neur.jpg"
image_len = 512
wavelet_len = 256

FOLDER = r'F:\PhD\Datasets\twitter-collection-master\twitter-collection-master\Final Dataset v11'


#cv2.imshow("Final Collage 1",collage1_resize)




if __name__ == "__main__":
#    print ("Executed when invoked directly")

    
    uid = "749003"
    imgid = '1'
    
    enc = get_dp_vit_enc(uid,imgid)
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(enc.shape)
    print(np.max(enc))
    print(np.min(enc))
#    print(enc[2120])
#    print(enc[100])
    











