from django.urls import path
from . import views

urlpatterns = [
    path('', views.search_view, name='search_view'),
    path('api/', views.search_api, name='search_api'),
]