from PIL import Image
import random
import os


def rotate_img(img):
    """
    Rotates an image by a random amount,
    :param img: The image to rotate
    :return: An image object, rotated by a random amount
    """
    return img.rotate(random.randint(-90, 90))


def move_pixels(img, pct):
    """
    Randomly moves pixels of an image
    :param img: The image to distort
    :param pct: The percentage likelihood of changing black pixels to white ones (0 to 100)
    :return: The distorted image
    """
    im = img.convert('RGB')
    pix = im.load()
    for i in range(im.size[0]):    # for every col:
        for j in range(im.size[1]):    # For every row
            r, g, b = im.getpixel((i, j))
            if r == 0 and g == 0 and b == 0:
                # print("found black pixel")
                if random.randint(0, 100) >= 100 - pct: # 10% chance to change a pixel
                    # print("should change col")
                    pix[i, j] = (255, 255, 255)
            # else:
                # print("found white pixel")

    return im




dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/distorted'
for filename in os.listdir("digits"):
    if filename.endswith('.png'):
        im = Image.open("digits/{}".format(filename))
        im = rotate_img(im)
        im = move_pixels(im, 1)
        # im.show()
        im.save("digits/distorted/{}".format(filename), 'PNG')




