# Generated by Django 3.2.5 on 2021-08-01 15:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('comp', '0003_auto_20210730_0932'),
    ]

    operations = [
        migrations.CreateModel(
            name='MaskImg2Img',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mask', models.ImageField(upload_to='', verbose_name='Mask')),
            ],
            options={
                'verbose_name': 'Маска',
                'verbose_name_plural': 'Маски',
            },
        ),
    ]
