from PIL import Image
import sys

image = Image.open(sys.argv[1])
pixels = list(image.getdata())
pixels = [(pixel[0]//2, pixel[1]//2, pixel[2]//2) for pixel in pixels]
image.putdata(pixels)
image.save('Q2.png')

