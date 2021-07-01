from PIL import Image
import io

with open("../dataset/cat/cat.1.jpg","rb") as file: 
    data = file.read() 
    img = Image.open(io.BytesIO(data)) 
    # img.show() 

print(type(img))