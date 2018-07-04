"""Process an image scan into an out directory, cropping and skewing per OpenCV magic"""
import os

import click
import numpy as np
import cv2

from utils import process_image

@click.command()
@click.argument('infile')
@click.argument('outdir')
def process(infile, outdir):
    """Process an image file to its relevant text block, writing the same named file to `outdir`"""
    with open(infile, 'rb') as img:
        outfile = os.path.join(outdir, os.path.basename(infile))
        image_array = np.asarray(bytearray(img.read()), dtype=np.uint8)
        cropped_image = process_image(image_array)
        cv2.imwrite(outfile, cropped_image)

