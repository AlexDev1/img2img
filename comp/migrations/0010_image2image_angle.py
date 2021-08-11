# Generated by Django 3.2.5 on 2021-08-11 11:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('comp', '0009_auto_20210811_1153'),
    ]

    operations = [
        migrations.AddField(
            model_name='image2image',
            name='angle',
            field=models.PositiveSmallIntegerField(choices=[(0, '0px'), (1, '1px'), (2, '2px'), (3, '3px'), (4, '4px'), (5, '5px'), (6, '6px'), (7, '7px'), (8, '8px'), (9, '9px'), (10, '10px')], default=0, verbose_name='Ширина линии'),
            preserve_default=False,
        ),
    ]