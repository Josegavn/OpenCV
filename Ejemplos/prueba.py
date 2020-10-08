from __future__ import print_function
import argparse
import cv2

argument_parser = argparse.ArgumentParser
argument_parser.add_argument('-i', '--image', required=True, help='C:\Josegavn\OpenVC\Ejemplos\Canela.jpg')
arguments = vars(argument_parser.parse_args())

image = cv2.imread(arguments['image'])

print(f'width: {image.shape[1] pixels')
print(f'heigth: {image.shape[0] pixels')
print(f'channels: {image.shape[2] channels')

cv2.imshow('Image', image)
cv2.waitKey(0)

cv2.imwrite('newimage.jpg', image)