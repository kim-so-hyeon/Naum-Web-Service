from webservice import views
from django.urls import path

### path url 하나 당 viwe fuction 하나만 사용 가능 ###

urlpatterns = [
    path("main/", views.main),
    # path('sub/', views.sub, name='sub'),
    path('sub/', views.sub, name='sub'),
    path('naverguide/', views.guide, name='guide'),
]


    
