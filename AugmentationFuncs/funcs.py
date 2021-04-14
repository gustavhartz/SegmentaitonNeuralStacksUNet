
import random
from skimage import filters,transform




def rotate(img,label):
    angle = random.randint(1,359)
    
    #Assume image is same x,y shape
    cmin = int(img.shape[0]/3)
    cmax = 2*int(img.shape[0]/3)
    center_x = random.randint(cmin,cmax)
    center_y = random.randint(cmin,cmax)
    
    return (transform.rotate(img,angle=angle,center=(center_x,center_y),mode='reflect'),
            transform.rotate(label,angle=angle,center=(center_x,center_y),mode='reflect'))