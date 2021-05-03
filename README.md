# Challenge: CAPTCHA-me-if-you-can
Por William Aguilar Restrepo

Para la solución de este reto, se implentan dos archivos de Python:
    1. Creación para el modelo de entrenamiento (**model.py**).
    2. Captura y envía la información de la imagen (**request_image.py**).
A continuación, se anexa la idea desarrolada para dar solución al reto **CATCH-me-if-you-can** de cada uno de los archivos.

Para el reto no se podía usar Pytesserac (herramienta usada para convertir los strings contenidas en la imagen y pasar el resultado en texto), por lo que se decide implementar el archivo **model.py**, y posteriormente, se crea el archivo request_image.py para obtener las imagenes y usar el modelo obtenido del archivo anterior para predecir la correcta calificación y enviar el string correcto.

## Creación para el modelo de entrenamiento
La idea de construir el archivo **model.py** surge como una respuesta para poder identificar las letras y números de la imagen obtenida del reto **CATCH-me-if-you-can**, en este archivo se implementa un modelo de entrenamiento usando Tensorflow para hacer de detección de letras, los pasos a seguir son los siguientes:

1. Se cargan las librerías de Tensorflow para construir el modelo, para crear el modelo, cargar el método train_test_split de scikit learn para subdividir la base de datos en entrenamiento y prueba, además de numpy y pandas para leer archivo y futuras operaciones.
 
```
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.python.framework import graph_io
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
print(tf.__version__)
```

2. Se carga el archivo emnist-mnist-train.csv obtenido de https://www.kaggle.com/crawford/emnist

```
train = pd.read_csv(r'C:\Users\William\Desktop\catch_me_if_you_can/emnist-mnist-train.csv')
``` 

3. Se definen la cantidad de clases, la etiqueta de las variables, las variables, el tamaño de las imagenes y modificar la forma de las variables cargadas del archivo.

```
classes = train.iloc[:,0].nunique()
x = np.array(train.iloc[:,1:].values)
y = np.array(train.iloc[:,0].values)
x = x/255

train_shape = train.shape[0]
train_height = 28
train_width = 28
train_size = train_height*train_width

x = x.reshape(train_shape, train_height, train_width, 1)
y = to_categorical(y, classes)
```

4. Se divide *x* y *y* para el entranamiento y evaluación del modelo.

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
```

5. Se construye el modelo con las capas correspondientes del modelo, usando una capa convolucional Conv2D de 32, un MaxPool de 2x2, en movimientos de 2, Flatten para aplanar el modelo, se completa con capas Dense de 512, 128 neuronas usando la función relu, y en el último una capa Dense que es el output de la cantidad de clases usando la función de activación softmax (categóricas > 2) y por último se compila el modelo usando ciertas métricas estadísticas.
Se agregan unas funciones de parada, cuando el modelo no pueda mejorar. 

```
stop = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights=True, patience=3,
                      mode='max')
reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, mil_lr=0.0001)
```

6. Se ajusta el modelo (puede tardar varios minutos).

```
history = model.fit(
    x_train, y_train, batch_size=64, epochs = 20, validation_data = (x_test, y_test),
    verbose = 1, steps_per_epoch=x_train.shape[0] // 64, callbacks=[reduce,stop])
```

7. Una vez que se tenga el modelo se guardan los pesos obtenidos y se vuelven a cargar (este paso es para posteriormente ser guardados como .pb o .pbtxt con mayor facilidad en dado caso que se requiera).

```
os.makedirs('C:\Users\William\Desktop\catch_me_if_you_can\model', exist_ok=True)
model.save('C:\Users\William\Desktop\catch_me_if_you_can\model\keras_model.h5')
model = load_model(r'C:\Users\William\Desktop\catch_me_if_you_can\model\keras_model.h5')
```

8. Por último se guardan los pesos como .pb o .pbtxt

```
tf.saved_model.save(model, r'C:\Users\William\Desktop\catch_me_if_you_can\model')
import sys
with tf.compat.v1.gfile.FastGFile(r"C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        proto_b = f.read()
        graph_def = tf.compat.v1.GraphDef()
        text_format.Merge(proto_b, graph_def) 
        _ = tf.graph_util.import_graph_def(graph_def, name='')
```

## Captura y envía la información de la imagen

Una vez se obtiene el modelo la idea es cargar el modelo usando cv2 (Open cv), obtener la imagen de la página web convertirla a texto y posteriormente enviar el string detectado a la pagina web.

1. Se cargan las librerías necesarias para usar Tensorflow Open-cv, y obtener la imagen del http.

```
import cv2
import requests
from bs4 import BeautifulSoup
import urllib
```

2. Se carga el modelo de Tensorflow en cv2

```
tensorflowNet = cv2.dnn.readNetFromTensorflow(
    r'C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pb',
    r'C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pbtxt')
```

3. Se carga la imagen de "http://challenge01.root-me.org/programmation/ch8/" y se hace el preprocesamiento requerido en la imagen usando cv2, como cambiar a escala de grises usar un kernel para limpiar el ruido de la imagen.

```
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
```

4. Usar el modelo de Tensorflow en cv2 para detectar los respectivos carácteres.

```
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(250, 50), swapRB=True, crop=False))
networkOutput = tensorflowNet.forward()
```

5. El último paso es detectar las correspondientes letras o números., para este paso quedo faltando agregar la funcion function_text(cropped) que permitiera anexar cada letra encontrada y devolverla como un texto, para esta parte lo que se haría es del modelo entrenado agregar las clases según la clase determinada por cv2.

```
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = im2[y:y + h, x:x + w]
    file = open(r"C:\Users\William\Desktop\recognized.txt", "a")
    text = function_text(cropped)
```

6. Una vez obtenido el texto, se hace un post http para arrojar el resultado obtenido del modelo pre-entrenado y lo que se encontró usando cv2.

```
requests.post(URL, data=text)
```

