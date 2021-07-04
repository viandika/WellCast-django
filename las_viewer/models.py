from django.conf import settings
from django.db import models
from hashid_field import HashidAutoField
from las_viewer.utils import validate_is_las


class LasUpload(models.Model):
    hashed_filename = HashidAutoField(primary_key=True)
    filename = models.CharField(max_length=50)
    las_file = models.FileField(
        upload_to=settings.MEDIA_ROOT / "las", validators=[validate_is_las]
    )

    # def save(self, *args, **kwargs):
    #     try:
    #         this = LasUpload.objects.get(hashed_filename=self.hashed_filename)
    #         if this.las_file:
    #             this.las_file.delete()
    #     except ObjectDoesNotExist:
    #         pass
    #     super(LasUpload, self).save(*args, **kwargs)


class ContactUs(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    message = models.TextField()
