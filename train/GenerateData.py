from concurrent.futures import thread
import cv2
from PIL import Image
from inspect import ismethod
import os
import numpy as np


# Get nya nanti dari upload-an file gambar
path = "static/uploads/Untitled-2-2.jpg"
path_save = "static/results/test/"
fname_save = "bri"


def run(obj, *args, **kwargs):
    for name in dir(obj):
        attribute = getattr(obj, name)
        if ismethod(attribute):
            attribute(*args, **kwargs)


def reformat_image(imgpath, filename_new, width_img, height_img, bg = 'white'):

    if (bg == 'white'):
        set_bg = (255,255,255,255)
    else:
        set_bg = (0,0,0,0)

    from PIL import Image
    # image = Image.open(imgpath, 'r')
    image =  Image.fromarray(imgpath)
    width = width_img
    height = height_img

    bigside = width if width > height else height

    background = Image.new(
        'RGBA', (bigside, bigside), set_bg)
    offset = (int(round(((bigside - width) / 2), 0)),
              int(round(((bigside - height) / 2), 0)))

    background.paste(image, offset)
    background.save(filename_new)

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255,255,255))


class GenerateData():  
    def resized_img(self, path, path_save, fname_save):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        height_img, width_img, channels = img.shape

        for i in range(5, 100):
            scale_percent = i
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            reformat_image(resized, path_save + 'data_'+fname_save + '_resized_' + str(i) + '.png', width_img, height_img)
            # print(resized)
            # cv2.imwrite(path_save + fname_save +
            #             '_resized_' + str(i) + '.jpg', resized)
            # reformat_image(path_save + fname_save +'_resized_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_resized_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_resized_' + str(i) + '.jpg')

    def resized_img_width(self, path, path_save, fname_save):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        height_img, width_img, channels = img.shape
        for i in range(5, 100):
            scale_percent = i
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0])
            dim = (width, height)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            reformat_image(resized, path_save + 'data_'+fname_save + '_resized_width_' + str(i) + '.png', width_img, height_img)

            # cv2.imwrite(path_save + fname_save +
            #             '_resized_width_' + str(i) + '.jpg', resized)
            # reformat_image(path_save + fname_save +
            #                '_resized_width_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_resized_width_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_resized_width_' + str(i) + '.jpg')

    def resized_img_height(self, path, path_save, fname_save):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        height_img, width_img, channels = img.shape
        for i in range(5, 100):
            scale_percent = i
            width = int(img.shape[1])
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            reformat_image(resized, path_save + 'data_'+fname_save + '_resized_height_' + str(i) + '.png', width_img, height_img)

            # cv2.imwrite(path_save + fname_save + '_resized_height_' +
            #             str(i) + '.jpg', resized)
            # reformat_image(path_save + fname_save +
            #                '_resized_height_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_resized_height_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_resized_height_' + str(i) + '.jpg')

    def rotate_img(self, path, path_save, fname_save):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
       
        for i in range(0, 360, 2):
            rotated = rotate_bound(img, i)
            height_img, width_img, channels = rotated.shape
            rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            grayImage = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            (thresh, bw_rotate) = cv2.threshold(
                grayImage, 100, 255, cv2.THRESH_BINARY)
            (thresh, wb_rotate) = cv2.threshold(
                grayImage, 100, 255, cv2.THRESH_BINARY_INV)
            reformat_image(rotated, path_save + 'data_'+fname_save + '_rotated_' + str(i) + '.png', width_img, height_img)
            reformat_image(bw_rotate, path_save + 'data_'+fname_save + '_bwrotated_' + str(i) + '.png', width_img, height_img)
            reformat_image(wb_rotate, path_save + 'data_'+fname_save + '_wbrotated_' + str(i) + '.png', width_img, height_img, 'black')

            # cv2.imwrite(path_save + fname_save +
            #             '_rotated_' + str(i) + '.jpg', rotated)
            # reformat_image(path_save + fname_save +
            #                '_rotated_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_rotated_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_rotated_' + str(i) + '.jpg')

    def bnw(self, path, path_save, fname_save):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height_img, width_img, channels = img.shape

        for i in range(0, 50):
            (thresh, bw) = cv2.threshold(
                grayImage, 100 + i, 255, cv2.THRESH_BINARY)
            (thresh, wb) = cv2.threshold(
                grayImage, 100 + i, 255, cv2.THRESH_BINARY_INV)
            reformat_image(wb, path_save + 'data_'+fname_save + '_wb_' + str(i) + '.png', width_img, height_img)
            reformat_image(bw, path_save + 'data_'+fname_save + '_bw_' + str(i) + '.png', width_img, height_img)

            # cv2.imwrite(path_save + fname_save +
            #             '_bw_' + str(i) + '.jpg', bw)
            # cv2.imwrite(path_save + fname_save +
            #             '_wb_' + str(i) + '.jpg', wb)

            # reformat_image(path_save + fname_save +
            #                '_bw_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_bw_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_bw_' + str(i) + '.jpg')

            # reformat_image(path_save + fname_save +
            #                '_wb_' + str(i) + '.jpg', path_save + 'data_'+fname_save + '_wb_' + str(i) + '.png', width_img, height_img)
            # os.remove(path_save + fname_save +
            #           '_wb_' + str(i) + '.jpg')

    def transparent(self, path, path_save, fname_save):
        img = Image.open(path)
        img = img.convert("RGBA")

        datas = img.getdata()

        nData = []

        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                nData.append((255, 255, 255, 0))
            else:
                nData.append(item)

        img.putdata(nData)
        for i in range(0, 20):
            img.save(path_save + fname_save +
                     "_trans_" + str(i) + '.png', "PNG")


if __name__ == "__main__":
    try:
        run(GenerateData(), path, path_save, fname_save)
        # cl = GenerateData()
        # cl.resized_img(path, path_save, fname_save)
    except Exception as e:
        print(e)
