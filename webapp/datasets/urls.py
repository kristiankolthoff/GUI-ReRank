from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataset_list, name='dataset_list'),
    path('add/', views.add_dataset, name='add_dataset'),
    path('<int:dataset_id>/search/', views.search_view, name='search_view'),
    path('<int:dataset_id>/edit/', views.edit_dataset, name='edit_dataset'),
    path('<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('api/list/', views.dataset_list_api, name='dataset_list_api'),
    path('set_active/', views.set_active_dataset, name='set_active_dataset'),
] 