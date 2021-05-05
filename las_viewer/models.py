from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from hashid_field import HashidAutoField


class LasUpload(models.Model):
    hashed_filename = HashidAutoField(primary_key=True)
    filename = models.CharField(max_length=50)
    las_file = models.FileField(upload_to='las/')

    # def save(self, *args, **kwargs):
    #     try:
    #         this = LasUpload.objects.get(hashed_filename=self.hashed_filename)
    #         if this.las_file:
    #             this.las_file.delete()
    #     except ObjectDoesNotExist:
    #         pass
    #     super(LasUpload, self).save(*args, **kwargs)
