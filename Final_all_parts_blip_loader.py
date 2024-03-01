from pathlib import Path
import numpy as np



df_csv_file = "Dataset_uniqid_class.csv"
parent_folder = r'E:\CVPR 24\datasets\Personality dataset output\default settings'

def get_all_parts_blip(sample_id):
#    print("got nothing")

#    directory = sample_label
#    new_dir = Path(FOLDER,directory)
    
    
    folder_name = sample_id
    filename = sample_id + '.npy'
    
#    folder_path = parent_folder / folder_name
#    filepath = folder_path / filename
#    
    filepath = Path(parent_folder,folder_name,filename)
    
    enc = np.load(filepath)
    return enc



if __name__ == "__main__":
#    print ("Executed when invoked directly")

    
#    uid = "222188599"
#    imgid = '48'
    sample_id = "222188599_48"
    
    enc = get_all_parts_blip(sample_id)
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(enc.shape)
    print(enc)
    print(np.max(enc))
    print(np.min(enc))



























