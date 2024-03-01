from transformers import AutoProcessor, BlipModel


model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

#texts = ["a photo of a cat riding a ball"]
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text = texts, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)

print(text_features.shape)
print(text_features)





