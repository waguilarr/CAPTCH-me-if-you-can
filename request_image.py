import cv2
import requests
from bs4 import BeautifulSoup
import urllib

tensorflowNet = cv2.dnn.readNetFromTensorflow(
    r'C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pb',
    r'C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pbtxt')

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
image_url = soup.find('img')['src']
urllib.request.urlretrieve(image_url, r'C:\Users\William\Desktop\imagen.jpeg')
img = cv2.imread(r'C:\Users\William\Desktop\imagen.jpeg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, rect_kernel)
img = opening.copy()
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(250, 50), swapRB=True, crop=False))
networkOutput = tensorflowNet.forward()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = im2[y:y + h, x:x + w]
    file = open(r"C:\Users\William\Desktop\recognized.txt", "a")
    text = function_text(cropped)

requests.post(URL, data=text)
