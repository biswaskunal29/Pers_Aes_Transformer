FOLDER = r'F:/PhD\Datasets/twitter-collection-master/twitter-collection-master/Final Dataset v11/'

def get_image_labels_tokens(uid,imgid):        #returns a list of all the labels
    filename = FOLDER + uid + "/" + imgid + "_labels_tokens.txt"  
#    print(filename)
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []

def get_profile_labels_tokens(uid,imgid):      #returns a list of all the labels
    filename = FOLDER + uid + "/" + uid + "_profile_labels_tokens.txt" 
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []

def get_banner_labels_tokens(uid,imgid):       #returns a list of all the labels
    filename = FOLDER + uid + "/" + uid + "_banner_labels_tokens.txt" 
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []

def get_image_HW_tokens(uid,imgid):        #returns a list of all the recognised handwriting(HW)
    filename = FOLDER + uid + "/" + imgid + "_paragraph_tokens.txt" 
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []

def get_profile_HW_tokens(uid,imgid):      #returns a list of all the recognised handwriting(HW)
    filename = FOLDER + uid + "/" + uid + "_profile_paragraph_tokens.txt"   
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []
    
def get_banner_HW_tokens(uid,imgid):      #returns a list of all the recognised handwriting(HW)
    filename = FOLDER + uid + "/" + uid + "_banner_paragraph_tokens.txt"     
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []

def get_image_desc_tokens(uid,imgid):      #returns a list of captions/image description written by the user about the image
    filename = FOLDER + uid + "/" + imgid + "_desc_tokens.txt"
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []
                
def get_profile_desc_tokens(uid,imgid):      #returns list of description written by the user on his profile
    filename = FOLDER + uid + "/" + uid + "_profiledata_tokens.txt"
    try:
        with open(filename, "r") as file:
            label_tokens = eval(file.readline())
        return label_tokens
    except:
        return []


#with open(list_file, "r") as file:
#    data2 = eval(file.readline())
       
# =============================================================================
# uid = "749003"
# imgid = "46"
#     
# result = get_image_desc_tokens(uid,imgid)
# print(result)
# =============================================================================
        
    
    