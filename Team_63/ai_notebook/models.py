from django.conf import settings
from django.db import models


# -------------------------
# NOTEBOOK MODEL
# -------------------------
class Notebook(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notebooks",
    )
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title


# -------------------------
# SOURCE MODEL (TEXT / URL / FILE)
# -------------------------
class Source(models.Model):

    # FIRST define constants
    TEXT = "text"
    URL = "url"
    FILE = "file"

    # THEN define choices
    SOURCE_TYPE_CHOICES = [
        (TEXT, "Text"),
        (URL, "URL"),
        (FILE, "File"),
    ]

    notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="sources",
    )

    title = models.CharField(max_length=200)

    source_type = models.CharField(
        max_length=10,
        choices=SOURCE_TYPE_CHOICES,
        default=TEXT,
    )

    # position for drag & drop ordering
    position = models.PositiveIntegerField(default=0)

    # URL sources
    url = models.URLField(blank=True, null=True)

    # Uploaded file
    file = models.FileField(
        upload_to="notebook_files/",
        blank=True,
        null=True,
    )

    # Extracted text or pasted text
    content = models.TextField(
        blank=True,
        null=True,
        help_text="Text content of this source (pasted OR extracted).",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["position", "-created_at"]

    def __str__(self):
        return f"{self.title} ({self.get_source_type_display()})"


# -------------------------
# CHAT MESSAGE MODEL
# -------------------------
class ChatMessage(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"

    ROLE_CHOICES = [
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
    ]

    notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:40]}..."
