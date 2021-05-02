from django.forms import ModelForm
from las_viewer.models import LasUpload


class LasUploadForm(ModelForm):
    class Meta:
        model = LasUpload
        fields = ['las_file']
