from PIL import Image, ImageStat
import requests
url = 'https://v3b.fal.media/files/b/0a946a30/1-YagS_kfug5vUSjRDJaA.jpg'
data = requests.get(url).content
with open('test_check.jpg', 'wb') as f: f.write(data)
img = Image.open('test_check.jpg').convert('RGB')
stat = ImageStat.Stat(img)
print('Extrema:', stat.extrema)
