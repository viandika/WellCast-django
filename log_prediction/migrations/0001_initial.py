# Generated by Django 3.2 on 2022-05-16 09:31

from django.db import migrations, models
import hashid_field.field
import log_prediction.utils
import pathlib


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ContactUs',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=50, null=True)),
                ('email', models.EmailField(blank=True, max_length=254, null=True)),
                ('message', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='LasUpload',
            fields=[
                ('hashed_filename', hashid_field.field.HashidAutoField(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', min_length=7, prefix='', primary_key=True, serialize=False)),
                ('filename', models.CharField(max_length=50)),
                ('las_file', models.FileField(upload_to=pathlib.PureWindowsPath('D:/codes/python/WellCast-django/media/las'), validators=[log_prediction.utils.validate_is_las])),
            ],
        ),
    ]