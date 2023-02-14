 
from django.contrib import admin
from django.urls import path
from ImageClassifier import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name="index"),
]
