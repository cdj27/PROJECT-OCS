"""Colour-independent paint evidence; bay geometry must still validate it."""
import cv2
import numpy as np


def paint_response(image,split_polarity=False):
    """Return local paint contrast, not a whitelist of marking hues.

    Bright neutral paint, locally distinct coloured paint, and dark neutral
    markings on light pavement are supported. Uniform coloured surfaces are
    not paint evidence. This is a proposal signal, never an occupancy label.
    """
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    light=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    saturation=hsv[:,:,1].astype(np.float32);value=hsv[:,:,2]
    sigma=max(4,min(image.shape[:2])*.018)
    background=cv2.GaussianBlur(light,(0,0),sigma)
    saturation_background=cv2.GaussianBlur(saturation,(0,0),sigma)
    contrast=light-background;colour_contrast=saturation-saturation_background
    neutral=(saturation<80)&(value>60)&(contrast>7)
    coloured=(saturation>65)&(value>55)&((np.abs(contrast)>9)|(colour_contrast>22))
    dark=(background>150)&(saturation<70)&(contrast<-22)
    strength=np.maximum(np.abs(contrast),np.maximum(colour_contrast,0)*.45)
    response=np.where(neutral|coloured|dark,np.clip(strength,0,255),0).astype(np.uint8)
    return (response,contrast<0) if split_polarity else response
