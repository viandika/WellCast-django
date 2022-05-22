from django.conf import settings
from django.db import models
from hashid_field import HashidAutoField
from log_prediction.utils import validate_is_las


class LasUpload(models.Model):
    hashed_filename = HashidAutoField(primary_key=True)
    filename = models.CharField(max_length=50)
    las_file = models.FileField(
        upload_to=settings.MEDIA_ROOT / "las", validators=[validate_is_las]
    )

class ContactUs(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    message = models.TextField()
