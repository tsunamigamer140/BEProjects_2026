from django.urls import path
from . import views

app_name = "ai_app"

urlpatterns = [
    path('chat/', views.chat_page, name="chat"),
    path('profile/', views.profile_page, name="profile"),
    path('subscription/', views.subscription_page, name="subscription"),
path("admin/subscriptions/", views.admin_subscriptions, name="admin_subscriptions"),
path("admin/subscriptions/approve/<int:pk>/", views.approve_subscription, name="approve_subscription"),
path("admin/subscriptions/reject/<int:pk>/", views.reject_subscription, name="reject_subscription"),


    # NEW ROUTES YOU MISSED
    path('new-chat/', views.new_chat, name="new_chat"),
    path('clear-chat/', views.clear_chat, name="clear_chat"),

    # Export routes
    path('chat/export/<str:fmt>/', views.export_chat, name="export_chat"),
    path("guest-login/", views.guest_login, name="guest_login"),

]
