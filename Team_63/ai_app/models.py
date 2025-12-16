from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    request_count = models.IntegerField(default=0)
    max_requests = models.IntegerField(default=30)  # registered users get 30

    def remaining(self):
        return self.max_requests - self.request_count

    @property
    def is_premium(self):
        """Check premium status from Subscription model."""
        if hasattr(self.user, "subscription"):
            return self.user.subscription.active
        return False


class GuestSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    request_count = models.IntegerField(default=0)
    max_requests = models.IntegerField(default=10)


class Chat(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=255, null=True, blank=True)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)


class Subscription(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    active = models.BooleanField(default=False)

    # store original payment screenshot (optional but useful)
    screenshot = models.ImageField(upload_to="payments/", blank=True, null=True)

    # store payment reference
    payment_ref = models.CharField(max_length=200, blank=True)

    # time tracking
    requested_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {'Premium' if self.active else 'Free'}"
