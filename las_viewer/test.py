import os
import tempfile

from django.conf import settings
from django.core.exceptions import ValidationError
from django.test import TestCase

from las_viewer.models import LasUpload


class LasUploadTest(TestCase):
    def setUp(self) -> None:
        tempfile.tempdir = os.path.join(settings.MEDIA_ROOT, "las")
        tf1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        self.file1 = tf1.name
        tf1.close()
        tf2 = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
        with open(tf2.name, "w") as f:
            f.write("something")
        # print(magic.from_file(tf2.name, mime=True))
        # print(tf2.read())
        tf2.close()
        self.file2 = tf2.name
        self.upload_png = LasUpload(filename="las1", las_file=self.file1)
        self.upload_txt = LasUpload(filename="las2", las_file=self.file2)

    def test_upload_image(self):
        #     self.upload_png.full_clean()
        self.assertRaises(ValidationError, self.upload_png.full_clean)

    def test_upload_txt(self):
        # self.upload_txt.full_clean()
        self.assertRaises(ValidationError, self.upload_txt.full_clean)

    def tearDown(self) -> None:
        os.remove(self.file1)
        os.remove(self.file2)
        las = LasUpload.objects.all()
        las.delete()
