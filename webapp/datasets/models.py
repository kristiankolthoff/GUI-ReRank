from django.db import models

from .media_config import DATASET_IMAGES_BASE_PATH, DATASET_ZIPS_BASE_PATH

# Create your models here.

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    zip_file = models.FileField(upload_to='dataset_zips/')
    # is_processing = models.BooleanField(default=False)
    STATE_CHOICES = [
        ('started', 'Started'),
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('failed', 'Failed'),
        ('success', 'Success'),
    ]
    state = models.CharField(max_length=16, choices=STATE_CHOICES, default='started')
    annotation_model_name = models.CharField(max_length=255, blank=True, null=True)
    embedding_model_type = models.CharField(max_length=255, blank=True, null=True)
    embedding_model_name = models.CharField(max_length=255, blank=True, null=True)
    batch_size = models.IntegerField(default=5)
    error_message = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class ImageEntry(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='images')
    name = models.CharField(max_length=255)
    file = models.ImageField(upload_to=DATASET_IMAGES_BASE_PATH+'/')
    annotation = models.TextField(blank=True, default='')  # JSON or dict as string for now

    def __str__(self):
        return self.name

class SearchDimension(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='search_dimensions')
    name = models.CharField(max_length=255)
    annotation_description = models.TextField()
    type = models.CharField(max_length=64, default='embedding')
    query_decomposition = models.BooleanField(default=True)
    negation = models.BooleanField(default=True)
    weight = models.FloatField(default=0.5)
    pos_weight = models.FloatField(default=5.0)
    neg_weight = models.FloatField(default=1.0)

    def __str__(self):
        return f"{self.name} ({self.dataset.name})"
