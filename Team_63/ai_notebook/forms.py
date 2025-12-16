# ai_notebook/forms.py
from django import forms
from .models import Notebook, Source


class NotebookForm(forms.ModelForm):
    class Meta:
        model = Notebook
        fields = ["title", "description"]


class SourceForm(forms.ModelForm):
    class Meta:
        model = Source
        fields = ["title", "source_type", "content", "url", "file"]
        widgets = {
            "content": forms.Textarea(attrs={"rows": 4}),
            "url": forms.URLInput(attrs={"placeholder": "https://example.com/article"}),
        }


class ChatForm(forms.Form):
    message = forms.CharField(
        label="Ask something",
        widget=forms.Textarea(
            attrs={
                "rows": 3,
                "placeholder": "Ask a question about your sources...",
            }
        ),
    )
