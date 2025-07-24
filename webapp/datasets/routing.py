from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/dataset_progress/(?P<dataset_id>\d+)/$', consumers.DatasetProgressConsumer.as_asgi()),
] 