# Register your models here.
from django import forms
from django.contrib import admin

from comp.models import ImagesMerge, MaskImg2Img, Image2Image


@admin.register(MaskImg2Img)
class MaskImg2ImgAdmin(admin.ModelAdmin):
    list_display = ['mask_tag_list']
    fields = ['mask', 'mask_tag']
    readonly_fields = ('mask_tag',)


class CustomChoiceField(forms.ModelChoiceField):

    def label_from_instance(self, obj):
        return obj.mask_tag_list()


class ImagesMergeAdminForm(forms.ModelForm):
    mask_file = CustomChoiceField(widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
                                  queryset=MaskImg2Img.objects.all())

    class Meta:
        model = ImagesMerge
        fields = ['im1', 'im2', 'mask_file', ]


@admin.register(ImagesMerge)
class ImagesMergeAdmin(admin.ModelAdmin):
    form = ImagesMergeAdminForm
    list_display = ['image_tag']
    fields = ['im1', 'im2', 'mask_file', 'mask_tag', 'image_tag']
    readonly_fields = ('image_tag', 'mask_tag')



@admin.register(Image2Image)
class Image2ImageAdmin(admin.ModelAdmin):
    list_display = ['image_tag']
    # fields = ['im1', 'im2', 'image_tag']
    readonly_fields = ('image_tag', )

    fields = [
        ('im1', 'im2'),
        ('color', 'line_width', 'angle'),
        'image_tag'
    ]

