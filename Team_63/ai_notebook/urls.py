from django.urls import path
from . import views

app_name = "ai_notebook"

urlpatterns = [
    path("notebooks/", views.notebook_list, name="notebook_list"),
    path("notebooks/<int:pk>/", views.notebook_detail, name="notebook_detail"),
    path("notebooks/<int:pk>/delete/", views.notebook_delete, name="notebook_delete"),
    path("notebooks/<int:pk>/export/", views.notebook_export, name="notebook_export"),

    path("sources/<int:pk>/delete/", views.source_delete, name="source_delete"),
    path("notebooks/<int:notebook_pk>/sources/reorder/", views.source_reorder, name="source_reorder"),
    path("notebook/<int:pk>/clear-chat/", views.clear_notebook_chat, name="clear_notebook_chat"),

]
