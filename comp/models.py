import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.core.files import File
from django.db import models
from django.utils.safestring import mark_safe
from numpy import size
from skimage.transform import rescale


def get_gradient_2d(start, stop, width, height, is_horizontal):
    """
     Функция, которая генерирует 2D, ndarray который увеличивается или уменьшается
     с равными интервалами в вертикальном или горизонтальном направлении.
    """
    if is_horizontal:
        result_np_tile = np.tile(np.linspace(start, stop, width), (height, 1))
        return result_np_tile
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)
    enm_zip_ar = enumerate(zip(start_list, stop_list, is_horizontal_list))
    for i, (start, stop, is_horizontal) in enm_zip_ar:
        item = get_gradient_2d(start, stop, width, height, is_horizontal)
        result[:, :, i] = item

    return result


_SIZE_IMAGE_NEWS = [1100, 585]


class MaskImg2Img(models.Model):
    mask = models.ImageField("Mask")

    class Meta:
        verbose_name = "Маска"
        verbose_name_plural = "Маски"

    def mask_tag_list(self):
        if self.mask:
            return mark_safe(f'<img height="20" src="{self.mask.url}" />')
        else:
            return '-'

    def mask_tag(self):
        if self.mask:
            return mark_safe(f'<img height="250" src="{self.mask.url}" />')
        else:
            return '-'

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        mask = Image.open(self.mask.file)
        mask = mask.convert('RGB')
        file = mask.resize(_SIZE_IMAGE_NEWS)
        blob_mask = BytesIO()
        file.save(blob_mask, 'JPEG')
        self.mask.save('mask_file.jpeg', File(blob_mask), save=False)

        return super(MaskImg2Img, self).save(force_insert=False, force_update=False, using=None, update_fields=None)


class ImagesMerge(models.Model):
    im1 = models.ImageField("Каритнка справо", upload_to='img2img', blank=False)
    # cropping_im1 = ImageRatioField('im1', '430x360')
    im2 = models.ImageField("Картинка слево", upload_to='img2img', blank=False)
    # cropping_im2 = ImageRatioField('im2', '430x360')
    image = models.ImageField("Результат", upload_to='img2img_result', blank=True, null=True)
    mask_file = models.ForeignKey("MaskImg2Img", verbose_name="Шаблон маски",
                                  blank=True, null=True, on_delete=models.SET_NULL)
    mask = models.ImageField("Маска:", blank=True, null=True)

    def mask_tag(self):
        if self.mask and not self.mask_file:
            return mark_safe(f'<img height="250" src="{self.mask.url}" />')
        else:
            return '-'

    def image_tag(self):
        if self.image:
            return mark_safe(f'<img height="100%" src="{self.image.url}" />')
        else:
            return '-'

    mask_tag.short_description = 'Маска: '
    mask_tag.allow_tags = True
    image_tag.short_description = 'Результат: '
    image_tag.allow_tags = True

    class Meta:
        verbose_name = 'Картинка 2в1'
        verbose_name_plural = "Картинки 2в1"

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):

        size = _SIZE_IMAGE_NEWS
        xsize, ysize = size
        from PIL import Image

        im1 = Image.open(self.im1.file)
        im2 = Image.open(self.im2.file)
        delta = 10

        # Изменяем размер обоих картинок на единый размер
        im1 = im1.resize(size)
        im2 = im2.resize(size)
        # Save to im1


        # im1.show()
        # im2.show()

        src1 = np.array(im1)
        src2 = np.array(im2)

        # # Создаем маску для перехода
        # array = get_gradient_3d(1100, 585, (0, 0, 0), (255, 255, 255), (True, True, True))
        # mask_g = Image.fromarray(np.uint8(array))
        # #mask_g.show()

        # # Save MAsk to File
        # blob_mask = BytesIO()
        # mask_g.save(blob_mask, 'JPEG')
        # self.mask.save('mask_file.jpeg', File(blob_mask), save=False)

        # mask1 = np.array(mask_g)
        # mask1 = mask1/255
        # dst = src1 * mask1 + src2 * (1 - mask1)
        # new_im = Image.fromarray(dst.astype(np.uint8))
        # # new_im.show()
        # # file_name = f'{uuid.uuid4()}.jpeg'

        # v.2

        if self.mask_file:
            # mask = cv2.imread(src1)
            mask = np.zeros_like(src1)
            array = cv2.rectangle(mask, (550, 585), (1100, 0), (255, 255, 255), thickness=10)
            # cv2.imshow('image window', array)
            # cv2.waitKey(0)
        else:
            mask = np.zeros_like(src1)
            array = cv2.rectangle(mask, (550, 0), (1100, 585), (255, 255, 255), thickness=10)
        mask_g = Image.fromarray(np.uint8(array))
        mask_g.show()
        mask1 = np.array(mask_g)
        mask1 = mask1 / 255
        dst = src1 * mask1 + src2 * (1 - mask1)
        new_im = Image.fromarray(dst.astype(np.uint8))

        # V.3
        # img_path=self.im1.path
        # img = cv2.imread(img_path)
        # dim = (256, 256)
        # resizedLena = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
        # X, Y = resizedLena, resizedLena
        # Fusion_Horizontal = np.hstack((resizedLena, Y, X))
        # Fusion_Vertical = np.vstack((resizedLena, X))
        # cv2.imshow('Fusion_Vertical using vstack', Fusion_Vertical)
        # cv2.waitKey(0)
        # # Methode 2: Using Numpy (contanate)
        # Fusion_Vertical   = np.concatenate((resizedLena, X, Y), axis=0)
        # Fusion_Horizontal = np.concatenate((resizedLena, X, Y), axis=1)
        # cv2.imshow("Fusion_Horizontal usung concatenate", Fusion_Horizontal)
        # cv2.waitKey(0)
        # # Methode 3: Using OpenCV (vconcat, hconcat)
        # Fusion_Vertical = cv2.vconcat([resizedLena, X, Y])
        # Fusion_Horizontal = cv2.hconcat([resizedLena, X, Y])
        #
        # cv2.imshow("Fusion_Horizontal Using hconcat", Fusion_Horizontal)
        # cv2.waitKey(0)

        # # Save MAsk to File
        # blob_mask = BytesIO()
        # mask_g.save(blob_mask, 'JPEG')
        # self.mask.save('mask_file.jpeg', File(blob_mask), save=False)
        # Save to ImageField
        if self.image:
            file_name = os.path.basename(self.image.name)
            if os.path.exists(self.image.path):
                result_deleted = os.remove(self.image.path)
        else:
            file_name = 'news_img_merge.jpeg'
        blob_file = BytesIO()
        new_im.save(blob_file, 'JPEG')
        self.image.save(file_name, File(blob_file), save=False)

        return super(ImagesMerge, self).save(force_insert=False, force_update=False, using=None, update_fields=None)

    def get_thumbnail_img(self):
        return mark_safe(f'<img src={self.im1.url}/>')


class Image2Image(models.Model):
    COLOR_CHOICES = [
        (1, 'Черный'),
        (125, 'Серый'),
        (255, 'Белый')
    ]
    WIDTH_LINE_CHOICES = [
        (0, '0px'),
        (1, '1px'),
        (2, '2px'),
        (3, '3px'),
        (4, '4px'),
        (5, '5px'),
        (6, '6px'),
        (7, '7px'),
        (8, '8px'),
        (9, '9px'),
        (10, '10px')
    ]
    ANGLE_CHOICES = [
        (2.1, '0'),
        (2.2, '10'),
        (2.3, '150'),
        (2.4, '3px'),
        (2.5, '4px'),
        (2.6, '5px'),
        (2.7, '6px'),
        (2.8, '7px'),
        (8, '8px'),
        (9, '9px'),
        (10, '10px')
    ]
    im1 = models.ImageField("Каритнка справо", upload_to='img2img', blank=False)
    # cropping_im1 = ImageRatioField('im1', '430x360')
    im2 = models.ImageField("Картинка слево", upload_to='img2img', blank=False)
    # cropping_im2 = ImageRatioField('im2', '430x360')
    image = models.ImageField("Результат", upload_to='img2img_result', blank=True, null=True)
    color = models.PositiveSmallIntegerField("Цвет линии", choices=COLOR_CHOICES)
    line_width = models.PositiveSmallIntegerField("Ширина линии", choices=WIDTH_LINE_CHOICES)
    angle = models.FloatField("Угод наклона", choices=ANGLE_CHOICES)

    class Meta:
        verbose_name = "Merge to 2 image"
        verbose_name_plural = "Merges to 2 images"

    def image_tag(self):
        if self.image:
            return mark_safe(f'<img height="100%" src="{self.image.url}" />')
        else:
            return '-'

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        from matplotlib import pyplot as plt
        # from skimage.transform import rescale
        import numpy as np
        size = _SIZE_IMAGE_NEWS
        xsize, ysize = size
        from PIL import Image

        # im1 = Image.open(self.im1.file)
        # im2 = Image.open(self.im2.file)
        # delta = 10

        # Изменяем размер обоих картинок на единый размер
        # im1 = im1.resize(size)
        # im2 = im2.resize(size)

        img0 = plt.imread(self.im1.file)
        img1 = plt.imread(self.im2.file)

        scale = img1.shape[0] / img0.shape[0]
        img0_rescaled = (rescale(img0, [scale, scale, 1])[:, :img1.shape[1], :] * 255).astype(np.uint8)

        # Line
        if self.color == 0: # Color
            combined = np.ones_like(img1)
        else:
            combined = np.ones_like(img1) * self.color
        angle = -np.pi / self.angle # Угод наклона
        lower_intersection = 0.4  # нижний ниженее пересечение
        line_width = self.line_width * 10  # Ширина линии

        y, x, _ = img1.shape

        yy, xx = np.mgrid[:y, :x]
        img0_positions = (xx - lower_intersection * x) * np.tan(angle) - line_width // 2 > (yy - y)
        img1_positions = (xx - lower_intersection * x) * np.tan(angle) + line_width // 2 < (yy - y)

        combined[img0_positions] = img0_rescaled[img0_positions]
        combined[img1_positions] = img1[img1_positions]
        new_im = Image.fromarray(combined, 'RGB')
        new_im = new_im.resize(size)
        # cv2.imshow("Fusion_Horizontal Using hconcat", combined)

        if self.image:
            file_name = os.path.basename(self.image.name)
            if os.path.exists(self.image.path):
                result_deleted = os.remove(self.image.path)
        else:
            file_name = 'news_img_merge.jpeg'
        blob_file = BytesIO()
        new_im.save(blob_file, 'JPEG')
        self.image.save(file_name, File(blob_file), save=False)
        return super(Image2Image, self).save(force_insert=False, force_update=False, using=None, update_fields=None)