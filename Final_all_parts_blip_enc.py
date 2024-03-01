import csv
from pathlib import Path
import numpy as np
#import torch
#import Final_token_reader as tr
#from sentence_transformers import SentenceTransformer


from Final_image_blip_enc import get_image_blip_enc
from Final_image_text_blip_enc import get_image_text_blip_enc
from Final_dp_blip_enc import get_dp_blip_enc
from Final_bann_blip_enc import get_bann_blip_enc
from Final_profile_text_blip_enc import get_profile_text_blip_enc


#model = SentenceTransformer('all-MiniLM-L6-v2')

def save_all_part_blip_enc(uid,imgid,new_dir):
    enc1 = get_image_blip_enc(uid,imgid)
    enc2 = get_image_text_blip_enc(uid,imgid)
    enc3 = get_dp_blip_enc(uid,imgid)
    enc4 = get_bann_blip_enc(uid,imgid)
    enc5 = get_profile_text_blip_enc(uid,imgid)
    
    enc1_arr = enc1.detach().cpu().numpy()
    enc2_arr = enc2.detach().cpu().numpy()
    enc3_arr = enc3.detach().cpu().numpy()
    enc4_arr = enc4.detach().cpu().numpy()
    enc5_arr = enc5.detach().cpu().numpy()
    
    enc = np.hstack((enc1_arr, enc2_arr, enc3_arr, enc4_arr, enc5_arr))
#    print(enc.shape)
#    print(enc.dtype)
#    print(type(enc))
#    print(new_dir)
    
    filename = uid + '_' + imgid + '.npy'
    filepath = new_dir / filename    
    
    np.save(filepath, enc)


df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'E:\CVPR 24\datasets\Personality dataset output\default settings'
#E:\CVPR 24\datasets\Personality dataset output\default settings


with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

for i in range(2):
#    print(data[i][0],data[i][1])
#    print(i)
    
    sample_label = data[i][0]
    class_label = data[i][1] 
#    print(sample_label)
#    print(class_label)
    
    
    parts = [x.strip() for x in sample_label.split("_")]
    (uid,imgid) = parts
#    print(uid)
#    print(imgid)
    
    directory = sample_label
    new_dir = Path(FOLDER,directory)
    new_dir.mkdir(parents=True, exist_ok=True)
    
    save_all_part_blip_enc(uid,imgid,new_dir)
    
    print(str(i) + "-" + str(5001)) 
    
    
    
    















#part1 = '_image_blip'
#part2 = '_image_text_blip'
#part3 = '_dp_blip'
#part4 = '_banner_blip'
#part5 = '_profile_text_blip'

#part2 = '_dp_ocr'
#part5 = '_dp_labels'
#part6 = '_banner_labels'
#part7 = '_image_desc'
#part8 = '_profile_desc'
    
    
#    save_image_blip_enc(uid,imgid,new_dir)
#    save_image_text_blip_enc(uid,imgid,new_dir)
#    save_dp_blip_enc(uid,imgid,new_dir)
#    save_banner_blip_enc(uid,imgid,new_dir)
#    save_profile_text_blip_enc(uid,imgid,new_dir)
    
#    print(str(i) + "//" + str(5001))   


#def save_image_blip_enc(uid,imgid,new_dir):
#    
#    enc = get_image_blip_enc(uid,imgid)
##    print(enc.shape)
##    print(enc.dtype)
#    enc_arr = enc.detach().cpu().numpy()
##    print(enc_arr.dtype)
##    print(type(enc_arr))
#    filename = uid + '_' + imgid + part1 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc_arr)
    
#def save_image_text_blip_enc(uid,imgid,new_dir):
#    
#    enc = get_image_text_blip_enc(uid,imgid)
##    print(enc.shape)
#    filename = uid + '_' + imgid + part2 + '.pt'
#    filepath = new_dir / filename
#    torch.save(filepath, enc)   
#    
#def save_dp_blip_enc(uid,imgid,new_dir):
#    
#    enc = get_dp_blip_enc(uid,imgid)
##    print(enc.shape)
#    filename = uid + '_' + imgid + part3 + '.pt'
#    filepath = new_dir / filename
#    torch.save(filepath, enc)  
#    
#def save_banner_blip_enc(uid,imgid,new_dir):
#    
#    enc = get_bann_blip_enc(uid,imgid)
##    print(enc.shape)
#    filename = uid + '_' + imgid + part4 + '.pt'
#    filepath = new_dir / filename
#    torch.save(filepath, enc)   
#    
#def save_profile_text_blip_enc(uid,imgid,new_dir):
#    
#    enc = get_profile_text_blip_enc(uid,imgid)
##    print(enc.shape)
#    filename = uid + '_' + imgid + part5 + '.pt'
#    filepath = new_dir / filename
#    torch.save(filepath, enc)     
    
# =============================================================================
#     image_ocr = tr.get_image_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_ocr_emb = model.encode(image_ocr)
# #    print(image_ocr_emb.shape)
#     filename = sample_label + part1 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_ocr_emb)
# =============================================================================

 
#    save_image_ocr(uid,imgid,new_dir)
#    save_dp_ocr(uid,imgid,new_dir)
#    save_banner_ocr(uid,imgid,new_dir)
#    
#    save_image_labels(uid,imgid,new_dir)    
#    save_dp_labels(uid,imgid,new_dir) 
#    save_banner_labels(uid,imgid,new_dir)    
#
#    save_image_desc(uid,imgid,new_dir) 
#    save_profile_desc(uid,imgid,new_dir) 
    

# =============================================================================
#     image_ocr = tr.get_image_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_ocr_emb = model.encode(image_ocr)
#     print(image_ocr_emb.shape)
#     filename = sample_label + part1 + '.npy'
#     filepath = new_dir / filename
#     
#     np.save(filepath, image_ocr_emb)
# =============================================================================
    

# =============================================================================
#     filename = sample_label + part3 + '.npy'
#     filepath = new_dir / filename
#     
#     np.save(filepath, np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [8, 9, 0]]))
# =============================================================================
      
#    a = np.load(filepath)
    
#    print(a)

# =============================================================================
# def save_image_ocr(uid,imgid,new_dir):
#     image_ocr = tr.get_image_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_ocr_emb = model.encode(image_ocr)
# #    print(image_ocr_emb.shape)
#     filename = sample_label + part1 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_ocr_emb)
# 
# def save_dp_ocr(uid,imgid,new_dir):
#     dp_ocr = tr.get_profile_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     dp_ocr_emb = model.encode(dp_ocr)
# #    print(dp_ocr_emb.shape)
#     filename = sample_label + part2 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, dp_ocr_emb)    
# 
# def save_banner_ocr(uid,imgid,new_dir):
#     banner_ocr = tr.get_banner_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     banner_ocr_emb = model.encode(banner_ocr)
# #    print(banner_ocr_emb.shape)
#     filename = sample_label + part3 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, banner_ocr_emb)  
# 
# def save_image_labels(uid,imgid,new_dir):
#     image_labels = tr.get_image_labels_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_labels_emb = model.encode(image_labels)
# #    print(image_labels_emb.shape)
#     filename = sample_label + part4 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_labels_emb) 
#     
# def save_dp_labels(uid,imgid,new_dir):
#     dp_labels = tr.get_profile_labels_tokens(uid,imgid)
# #    print(len(image_ocr))
#     dp_labels_emb = model.encode(dp_labels)
# #    print(dp_labels_emb.shape)
#     filename = sample_label + part5 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, dp_labels_emb)   
#     
# def save_banner_labels(uid,imgid,new_dir):
#     banner_labels = tr.get_banner_labels_tokens(uid,imgid)
# #    print(len(image_ocr))
#     banner_labels_emb = model.encode(banner_labels)
# #    print(banner_labels_emb.shape)
#     filename = sample_label + part6 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, banner_labels_emb) 
#    
# def save_image_desc(uid,imgid,new_dir):
#     image_desc = tr.get_image_desc_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_desc_emb = model.encode(image_desc)
# #    print(image_desc_emb.shape)
#     filename = sample_label + part7 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_desc_emb)    
#     
# def save_profile_desc(uid,imgid,new_dir):
#     profile_desc = tr.get_profile_desc_tokens(uid,imgid)
# #    print(len(image_ocr))
#     profile_desc_emb = model.encode(profile_desc)
# #    print(profile_desc_emb.shape)
#     filename = sample_label + part8 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, profile_desc_emb) 
# =============================================================================   










