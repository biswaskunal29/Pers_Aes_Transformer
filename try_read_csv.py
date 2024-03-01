import csv



# =============================================================================
# # opening the CSV file
# with open('Dataset_uniqid_class.csv', mode ='r')as file:
#    
#   # reading the CSV file
#   csvFile = csv.reader(file)
#  
#   # displaying the contents of the CSV file
#   for lines in csvFile:
#         print(lines)
#         break
# =============================================================================

# =============================================================================
# with open("Dataset_uniqid_class.csv", "r") as imagelist_file:
#     linelist = imagelist_file.readlines()
# 
# #Now Downloading
# parts = [x.strip() for x in linelist[target_line].split(",")]
# (uid,extracted) = parts
# 
# if linelist[target_line].startswith("#"): continue
# if(extracted == "processed" or extracted == "failed process"): 
# #        print("Done")
#     continue
# 
# =============================================================================

#traitmap_rev = {0:'O', 1:'C', 2:'E', 3:'A', 4:'N'}
traitmap = {'O':0 ,'C':1 ,'E':2 ,'A':3 ,'N':4}

def read_dataset_to_list():
    X_list = []
    y_list = []
    with open("Dataset_uniqid_class.csv", newline='\n', encoding='utf-8') as csv_file:
        cf = csv.reader(csv_file, delimiter=',',quotechar='"')

        for row in cf:
            if(row[0]=='user_id'): continue
            X_list.append(row[0])
            y_list.append(traitmap[row[1]])
            
#            p_id = row[0]
#            class_id = traitmap[row[1]]        
#            ele = [p_id,class_id]
#            prof_list.append(ele)
    return X_list, y_list

x,y = read_dataset_to_list()


print(len(x))
print(len(y))
print(x[0])
print(type(x[0]))
print(y[0])
print(type(y[0]))

print(x.shape[0])

#print(len(profile_list))
#print(profile_list[0])
#print(profile_list[0][0])
#print(type(profile_list[0][0]))
#
#print(profile_list[0][1])
#print(type(profile_list[0][1]))




























