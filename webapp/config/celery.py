import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('gui_rerank_webapp')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()