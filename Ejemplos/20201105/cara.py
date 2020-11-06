#importar la vase de conocimietos de la aplicación
import numpy as np
import matplotlib.pyplot as plt
import cv2

cascada = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#función detectar rostro
def detecta_cara(imagen):
    imagen1 = imagen.copy()
    rectangulos = cascada.detectMultiScale(imagen1)
    for(x,y,w,h) in rectangulos:
        cv2.rectangle (imagen1, (x,y), (x+w, y+h), (0,255,0), 10)
        return imagen1


captura = cv2.VideoCapture(0)

#cierra hasta QUE!
while True:
    res, video = captura.read(0)
    video = detecta_cara(video)
    cv2.imshow('detectar en camara', video)

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

#cerrar
captura.release()
cv2.destroyAllWindows()