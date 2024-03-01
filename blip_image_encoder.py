from transformers import AutoProcessor, BlipModel
from PIL import Image
#import requests

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("222188599_48.jpg")

#image = Image.open("Original_Image.jpg") 

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)

print(image_features.shape)
print(type(image_features))
print(image_features.dtype)


























