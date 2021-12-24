# Python imports
from base64 import b64encode
from io import BytesIO
from os.path import join
from posixpath import abspath

# Django imports
from django import template
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView, View
from django.views.generic.edit import FormMixin

# external imports
import PIL.Image as Image

# app imports
from .forms import ImageForm
from .singlemotiondetector import SingleMotionDetector

register = template.Library()


class MainView(TemplateView):
    template_name = "index.html"


class DetectorView(FormView):
    form_class = ImageForm
    template_name = "detecator.html"

    def form_valid(self, form) -> HttpResponse:
        cleaned_data = form.cleaned_data

        name = cleaned_data["name"]

        age = cleaned_data["age"]
        emotion = cleaned_data["emotion"]
        agenda = cleaned_data["agenda"]

        print(age, emotion, agenda)

        image_name = cleaned_data["image"].name
        img_file = cleaned_data["image"].file

        file_binary = BytesIO.read(img_file)
        image = Image.open(BytesIO(file_binary))
        image.save("static/media/" + image_name)

        image_name = "static/media/" + image_name
        driver = SingleMotionDetector(image_name, True, False, False, False)
        driver_output = driver()

        print(driver)

        context = self.get_context_data()
        context["image"] = "/".join(driver_output.split("/")[1:])

        return render(self.request, self.template_name, context)

    def get_success_url(self) -> str:
        return reverse("emotion_detection:main")

    @register.filter
    def bin_2_img(self, _bin):
        if _bin is not None:
            return b64encode(_bin).decode("utf-8")
