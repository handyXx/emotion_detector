# Django imports
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, re_path

urlpatterns = [
    re_path(
        r"^login/$",
        auth_views.LoginView.as_view(template_name="core/login.html"),
        name="core_login",
    ),
    re_path(r"^logout/$", auth_views.LogoutView.as_view(), name="core_logout"),
    # enable the admin interface
    re_path(r"^admin/", admin.site.urls),
    re_path(r"", include("apps.emotion_detection.urls")),
]
