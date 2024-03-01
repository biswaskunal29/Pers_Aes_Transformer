import csv
from pathlib import Path
import numpy as np
#import Final_token_reader as tr
#from sentence_transformers import SentenceTransformer


from Final_image_vit_encoder import get_image_vit_enc
from Final_image_Text_encoder import get_image_text_enc
from Final_dp_vit_encoder import get_dp_vit_enc
from Final_banner_vit_encoder import get_banner_vit_enc
from Final_profile_Text_encoder import get_profile_text_enc


#model = SentenceTransformer('all-MiniLM-L6-v2')

def save_image_vit_enc(uid,imgid,new_dir):
    
    enc = get_image_vit_enc(uid,imgid)
#    print(enc.shape)
    filename = uid + '_' + imgid + part1 + '.npy'
    filepath = new_dir / filename
    np.save(filepath, enc)
    
def save_image_text_enc(uid,imgid,new_dir):
    
    enc = get_image_text_enc(uid,imgid, infobert = 0)
#    print(enc.shape)
    filename = uid + '_' + imgid + part2 + '.npy'
    filepath = new_dir / filename
    np.save(filepath, enc)   
    
def save_dp_vit_enc(uid,imgid,new_dir):
    
    enc = get_dp_vit_enc(uid,imgid)
#    print(enc.shape)
    filename = uid + '_' + imgid + part3 + '.npy'
    filepath = new_dir / filename
    np.save(filepath, enc)  
    
def save_banner_vit_enc(uid,imgid,new_dir):
    
    enc = get_banner_vit_enc(uid,imgid)
#    print(enc.shape)
    filename = uid + '_' + imgid + part4 + '.npy'
    filepath = new_dir / filename
    np.save(filepath, enc)   
    
def save_profile_text_enc(uid,imgid,new_dir):
    
    enc = get_profile_text_enc(uid,imgid, infobert = 0)
#    print(enc.shape)
    filename = uid + '_' + imgid + part5 + '.npy'
    filepath = new_dir / filename
    np.save(filepath, enc)     
    

# =============================================================================
#     image_ocr = tr.get_image_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_ocr_emb = model.encode(image_ocr)
# #    print(image_ocr_emb.shape)
#     filename = sample_label + part1 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_ocr_emb)
# =============================================================================


df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\Ablation 1 Infobert off'

part1 = '_image_vit'
part2 = '_image_text'
part3 = '_dp_vit'
part4 = '_banner_vit'
part5 = '_profile_text'

#part2 = '_dp_ocr'
#part5 = '_dp_labels'
#part6 = '_banner_labels'
#part7 = '_image_desc'
#part8 = '_profile_desc'


with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

for i in range(5002):
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
    
    save_image_vit_enc(uid,imgid,new_dir)
    save_image_text_enc(uid,imgid,new_dir)
    save_dp_vit_enc(uid,imgid,new_dir)
    save_banner_vit_enc(uid,imgid,new_dir)
    save_profile_text_enc(uid,imgid,new_dir)
    
    print(str(i) + "//" + str(5001))   
    
    
    
    
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

