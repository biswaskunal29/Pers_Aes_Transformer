#from pathlib import Path
import numpy as np
from Final_all_parts_blip_loader import get_all_parts_blip
from numpy.linalg import norm

def cosine_sim(A,B):
    if ((norm(A) == 0) or (norm(B) == 0)):
        return 0.0
    else:
        cosine = np.dot(A,B)/(norm(A)*norm(B))
        return cosine

def add_1_gram_sim(arr):
#    print(arr)
    arr_512 = np.reshape(arr, (-1,512))
    
#    print(arr_512.shape)
    sim_arr = np.empty((0, 513), dtype=np.float32)
    
    
    for single_enc in arr_512:
        sim = cosine_sim(single_enc,single_enc)
        single_enc = np.hstack((sim,single_enc))
#        print(single_enc.shape)
        sim_arr = np.vstack((sim_arr,single_enc))
        
        
#        break
#    print(sim_arr.shape)
    
#    print()
    return sim_arr

def add_2_gram_sim(arr):
    
    arr_1024 = np.reshape(arr, (-1,1024))
    sim_arr = np.empty((0, 1025), dtype=np.float32)
    
    for single_enc in arr_1024:
        
        sim = cosine_sim(single_enc[0:512],single_enc[512:1024])
        
#        sim = cosine_sim(single_enc,single_enc)
        single_enc = np.hstack((sim,single_enc))
#        print(single_enc.shape)
        sim_arr = np.vstack((sim_arr,single_enc))
        
        
#        break
    print(sim_arr.shape)
    
    print()

    return sim_arr





if __name__ == "__main__":
#    print ("Executed when invoked directly")

    
#    uid = "222188599"
#    imgid = '48'
    sample_id = "222188599_48"
    
    enc = get_all_parts_blip(sample_id)
    reshaped_arr = add_1_gram_sim(enc)
    reshaped_arr2_gram = add_2_gram_sim(enc)
    
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(reshaped_arr.shape)
    print(reshaped_arr2_gram.shape)
    
##    print(reshaped_arr)
    print(np.max(reshaped_arr2_gram))
    print(np.min(reshaped_arr2_gram))
    
#    print(reshaped_arr[0].shape)
#    print(reshaped_arr[30].shape)
#    
#    print(cosine_sim(reshaped_arr[0],reshaped_arr[0]))
#    print(cosine_sim(reshaped_arr[0],reshaped_arr[1]))
#    print(cosine_sim(reshaped_arr[2],reshaped_arr[30]))
    

#    a = np.array([.1, .1, .3, .4, .5])
#    b = np.array([.0, .0, .3, .5, .2])
#    
#    print(norm(b))
#    print(a[0:2].shape, a[2:4].shape)
#    print(cosine_sim(a[0:2],a[2:4]))




















