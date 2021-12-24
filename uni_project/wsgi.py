# Python imports
import os

# Django imports
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "uni_project.settings.development")
application = get_wsgi_application()
