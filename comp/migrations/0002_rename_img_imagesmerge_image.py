# Generated by Django 3.2.5 on 2021-07-30 08:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('comp', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imagesmerge',
            old_name='img',
            new_name='image',
        ),
    ]
