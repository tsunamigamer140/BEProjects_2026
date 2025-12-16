from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from ai_app import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Auth Pages
    path('', views.login_view, name="login"),
    path('register/', views.register_view, name="register"),
    path('logout/', views.logout_view, name="logout"),

    # Chat API Endpoint
    path('rag-chat-api/', views.rag_chat_api, name="rag_chat_api"),

    # App Routes (DO NOT repeat '' path here)
    path('app/', include('ai_app.urls')),   # changed for safety

    # Notebook App
    path("notebooks/", include("ai_notebook.urls", namespace="ai_notebook")),


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
