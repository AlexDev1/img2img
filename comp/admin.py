from django.contrib import admin

# Register your models here.
from comp.models import ImagesMerge


@admin.register(ImagesMerge)
class ImagesMergeAdmin(admin.ModelAdmin):
    list_display = ['image_tag']
    fields = ['im1', 'im2', 'mask_tag', 'image_tag']
    readonly_fields = ('image_tag', 'mask_tag')


