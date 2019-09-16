import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm

def susan_mask():
    mask=np.ones((7,7))
    mask[0,0]=0
    mask[0,1]=0
    mask[0,5]=0
    mask[0,6]=0
    mask[1,0]=0
    mask[1,6]=0
    mask[5,0]=0
    mask[5,6]=0
    mask[6,0]=0
    mask[6,1]=0
    mask[6,5]=0
    mask[6,6]=0
    return mask

def plot_image(image,title):
	plt.figure()

	plt.title(title)
	plt.imshow(image,cmap = 'gray')

	plt.show()


def susan_corner_detection(img):
    img = img.astype(np.float64)
    g=37/2
    circularMask=susan_mask()
    output=np.zeros(img.shape)

    for i in range(3,img.shape[0]-3):
        for j in range(3,img.shape[1]-3):
            ir=np.array(img[i-3:i+4, j-3:j+4])   # Captura de la zona de interéz dentro de la imagen
            ir =  ir[circularMask==1]    # Captura solo de los pixeles dentro de la mascara
            ir0 = img[i,j]    # Pixel nucleo
            a=np.sum(np.exp(-((ir-ir0)/10)**6))   # Calculo del area SUSAN respecto al nucleo
            if a<=g:                              # Comparación con límite
                a=g-a             
            else:
                a=0
            output[i,j]=a                         # Si es una esquina guarda un valor Diferente a 0
    return output

img=cv.imread("susan_input1.png",0)
plot_image(img, "Input Image")  # good success
output1 = susan_corner_detection(img)
finaloutput1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
finaloutput1[output1 != 0] = [255, 255, 0]
plot_image(finaloutput1, "Output Part1")  # good success
