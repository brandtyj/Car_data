import base64
import requests
from io import BytesIO
from PIL import Image


url = 'http://localhost:3000/infer'
r = requests.post(url,json={'class':1.8})
print(r.json())
#with open("test_image.jpg", "rb") as image_file:
#    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

#decoded_image_data = base64.b64decode(encoded_string)
#image_stream = BytesIO(decoded_image_data)
#image = Image.open(image_stream)#

#with open('output.txt', 'w') as f:
#    f.write(encoded_string)
#print(encoded_string)

