# Django imports
from django import forms


class ImageForm(forms.Form):
    name = forms.CharField(max_length=15)
    image = forms.ImageField()
    emotion = forms.BooleanField(required=False)
    age = forms.BooleanField(required=False)
    agenda = forms.BooleanField(required=False)
