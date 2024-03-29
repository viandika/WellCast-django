# Generated by Django 3.2 on 2021-07-03 03:08

from django.db import migrations, models
import las_viewer.utils


class Migration(migrations.Migration):

    dependencies = [
        ('las_viewer', '0001_initial'),
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
        migrations.AlterField(
            model_name='lasupload',
            name='las_file',
            field=models.FileField(upload_to='las/', validators=[las_viewer.utils.validate_is_las]),
        ),
    ]
