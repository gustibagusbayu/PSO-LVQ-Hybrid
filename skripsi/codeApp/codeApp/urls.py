from django.contrib import admin
from django.urls import path
from lvqPso.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', index, name='index'),
    path('test/', test, name='test'),
    path('user/', user, name='user'),
]
