import pickle
import Final_token_reader as tr
#import csv
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer


def get_profile_text_enc(uid,imgid, infobert = 1):
# =============================================================================
# IF infobert = 1 then we use info bert sorting
# IF infobert = 1 then we use normal bert 
# =============================================================================

#    image_ocr = tr.get_image_HW_tokens(uid,imgid)
    dp_ocr = tr.get_profile_HW_tokens(uid,imgid)
    banner_ocr = tr.get_banner_HW_tokens(uid,imgid)
    
#    image_labels = tr.get_image_labels_tokens(uid,imgid)
    dp_labels = tr.get_profile_labels_tokens(uid,imgid)
    banner_labels = tr.get_banner_labels_tokens(uid,imgid)
    
#    image_desc = tr.get_image_desc_tokens(uid,imgid)  
    profile_desc = tr.get_profile_desc_tokens(uid,imgid)
    
    str_list = []
#    str_list.extend(image_ocr)
    str_list.extend(dp_ocr)
    str_list.extend(banner_ocr)
    
#    str_list.extend(image_labels)
    str_list.extend(dp_labels)
    str_list.extend(banner_labels)
    
#    str_list.extend(image_desc) 
    str_list.extend(profile_desc)   
    
#    word_doc = ' '.join(str_list)
    
    if (infobert == 0):
        sorted_doc = str_list
    else:
        sorted_doc = sorted(str_list, reverse = True, key = word_imp)
    
    #    take only top profile_average_words words
    topn_sorted_doc = sorted_doc[:profile_average_words]
    
    encoded_doc = np.zeros((0,384),dtype = np.single)

    for i in range(len(topn_sorted_doc)):
#        print(topn_sorted_doc[i])
#        break
        i_emb = model.encode(topn_sorted_doc[i])
        encoded_doc = np.vstack((encoded_doc,i_emb))
    
#    print('before padding size ',encoded_doc.shape)
    
    padsize = profile_average_words - len(topn_sorted_doc)
    if( padsize > 0):
        pad_array = np.zeros((padsize,384),dtype = np.single)
#        print(pad_array.shape)
        encoded_doc = np.vstack((encoded_doc,pad_array))
    
#    print(encoded_doc.shape, encoded_doc[14][0] ,'\n')    
    
    encoded_doc = encoded_doc.flatten() 
    
    return encoded_doc

#def 

word_info_file = "word_info.pkl"
profile_average_words = 25

#read the dict of word information gain
with open(word_info_file, 'rb') as handle:
  word_info = pickle.loads(handle.read())

def_word_info = defaultdict(np.float64,word_info)

def word_imp(word):  #return the words mutual information
    imp = def_word_info[word]
    return imp

model = SentenceTransformer('all-MiniLM-L6-v2')



if __name__ == "__main__":
#    print ("Executed when invoked directly")

    uid = "749003"
    imgid = '1'

    enc = get_profile_text_enc(uid,imgid)
#    enc = image_encoding("749003_1.jpg", infobert = 1)


    print(enc.shape)
    print(np.max(enc))
    print(np.min(enc))
    
#    print(enc[1])
#    print(enc[1060])
#    print(enc[2120])






























