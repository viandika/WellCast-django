from wellcast.models import LasUpload, ContactUs
from django import forms


class LasUploadForm(forms.ModelForm):
    class Meta:
        model = LasUpload
        fields = ["las_file"]


class ContactUsForm(forms.ModelForm):
    name = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"class": "form-control mb-2"})
    )
    email = forms.EmailField(
        required=False, widget=forms.EmailInput(attrs={"class": "form-control mb-2"})
    )
    message = forms.CharField(
        widget=forms.Textarea(attrs={"class": "form-control mb-2"})
    )

    class Meta:
        model = ContactUs
        fields = ["name", "email", "message"]
