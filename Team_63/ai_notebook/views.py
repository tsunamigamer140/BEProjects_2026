from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render, redirect
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse

from django.db.models import Max

from .models import Notebook, Source, ChatMessage
from .forms import NotebookForm, SourceForm, ChatForm
from .services import generate_reply

import requests
from bs4 import BeautifulSoup
import pdfplumber
import docx
from PyPDF2 import PdfReader
import json


def clear_notebook_chat(request, pk):
    notebook = get_object_or_404(Notebook, pk=pk, owner=request.user)

    # delete all chat messages linked to this notebook
    ChatMessage.objects.filter(notebook=notebook).delete()

    return redirect("ai_notebook:notebook_detail", pk=pk)

# =========================================================
# Extraction Helpers
# =========================================================

def extract_text_from_pdf(file_field):
    try:
        text = ""
        file_field.open("rb")
        with pdfplumber.open(file_field) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                text += extracted + "\n"
        file_field.close()

        if text.strip():
            return text.strip()

        # Fallback → PyPDF2
        file_field.open("rb")
        reader = PdfReader(file_field)
        fallback = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            fallback += extracted + "\n"
        file_field.close()

        return fallback.strip() or "Could not extract text from PDF."

    except Exception as e:
        return f"PDF extraction failed: {e}"


def extract_text_from_docx(file_field):
    try:
        file_field.open("rb")
        d = docx.Document(file_field)
        text = "\n".join(p.text for p in d.paragraphs)
        file_field.close()
        return text.strip() or "Could not extract text from DOCX."
    except Exception as e:
        return f"DOCX extraction failed: {e}"


def extract_text_from_txt(file_field):
    try:
        file_field.open("rb")
        data = file_field.read()
        file_field.close()
        return data.decode("utf-8", errors="ignore") or "File was empty."
    except Exception as e:
        return f"TXT extraction failed: {e}"


def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text("\n")
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return cleaned[:50000] or "No readable text found."
    except Exception as e:
        return f"URL extraction failed: {e}"


# =========================================================
# Notebook List + Search
# =========================================================

@login_required
def notebook_list(request):
    q = request.GET.get("q", "").strip()

    notebooks = Notebook.objects.filter(owner=request.user)
    if q:
        notebooks = notebooks.filter(title__icontains=q)

    form = NotebookForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        nb = form.save(commit=False)
        nb.owner = request.user
        nb.save()
        return redirect("ai_notebook:notebook_detail", pk=nb.pk)

    return render(
        request,
        "ai_notebook/notebook_list.html",
        {"notebooks": notebooks, "form": form, "q": q},
    )


# =========================================================
# Notebook Detail
# =========================================================

@login_required
def notebook_detail(request, pk):
    notebook = get_object_or_404(Notebook, pk=pk, owner=request.user)

    source_form = SourceForm()
    chat_form = ChatForm()

    if request.method == "POST":

        # -------------------------------------------------
        # ADD SOURCE
        # -------------------------------------------------
        if "add_source" in request.POST:
            source_form = SourceForm(request.POST, request.FILES)

            if source_form.is_valid():
                src = source_form.save(commit=False)
                src.notebook = notebook

                # assign position (append at end)
                max_pos = notebook.sources.aggregate(m=Max("position"))["m"] or 0
                src.position = max_pos + 1

                # FIRST SAVE -> upload file to disk
                src.save()

                # TEXT
                if src.source_type == Source.TEXT:
                    if not src.content:
                        src.content = ""

                # URL
                elif src.source_type == Source.URL:
                    url_value = source_form.cleaned_data.get("url") or ""
                    src.url = url_value
                    src.content = extract_text_from_url(url_value)

                # FILE
                elif src.source_type == Source.FILE and src.file:
                    filepath = src.file.path.lower()

                    if filepath.endswith(".pdf"):
                        src.content = extract_text_from_pdf(src.file)
                    elif filepath.endswith(".docx"):
                        src.content = extract_text_from_docx(src.file)
                    elif filepath.endswith(".txt"):
                        src.content = extract_text_from_txt(src.file)
                    else:
                        src.content = "Unsupported file format. Upload PDF, DOCX, or TXT."

                src.save()
                return redirect("ai_notebook:notebook_detail", pk=notebook.pk)

        # -------------------------------------------------
        # EDIT SOURCE (from modal)
        # -------------------------------------------------
        elif "edit_source" in request.POST:
            source_id = request.POST.get("source_id")
            src = get_object_or_404(Source, pk=source_id, notebook=notebook)

            src.title = request.POST.get("title", src.title)
            src_type = request.POST.get("source_type", src.source_type)
            src.source_type = src_type

            if src_type == Source.TEXT:
                src.content = request.POST.get("content", "")
                src.url = ""

            elif src_type == Source.URL:
                url_value = request.POST.get("url", "")
                src.url = url_value
                src.content = extract_text_from_url(url_value)

            # file editing: keep existing file (simpler UX)
            src.save()
            return redirect("ai_notebook:notebook_detail", pk=notebook.pk)

        # -------------------------------------------------
        # SEND CHAT MESSAGE
        # -------------------------------------------------
        elif "send_message" in request.POST:
            chat_form = ChatForm(request.POST)

            if chat_form.is_valid():
                user_message = chat_form.cleaned_data["message"]

                ChatMessage.objects.create(
                    notebook=notebook,
                    role=ChatMessage.ROLE_USER,
                    content=user_message,
                )

                try:
                    assistant_reply = generate_reply(notebook, user_message)
                except Exception as e:
                    assistant_reply = f"AI Error: {e}"

                ChatMessage.objects.create(
                    notebook=notebook,
                    role=ChatMessage.ROLE_ASSISTANT,
                    content=assistant_reply,
                )

                return redirect("ai_notebook:notebook_detail", pk=notebook.pk)

    return render(
        request,
        "ai_notebook/notebook_detail.html",
        {
            "notebook": notebook,
            "sources": notebook.sources.all(),
            "messages": notebook.messages.all(),
            "source_form": source_form,
            "chat_form": chat_form,
        },
    )


# =========================================================
# Delete Notebook / Source
# =========================================================

@login_required
@require_POST
def notebook_delete(request, pk):
    notebook = get_object_or_404(Notebook, pk=pk, owner=request.user)
    notebook.delete()
    return redirect("ai_notebook:notebook_list")


@login_required
@require_POST
def source_delete(request, pk):
    src = get_object_or_404(Source, pk=pk, notebook__owner=request.user)
    notebook_pk = src.notebook.pk
    src.delete()
    return redirect("ai_notebook:notebook_detail", pk=notebook_pk)


# =========================================================
# Drag & Drop reorder
# =========================================================

@login_required
@require_POST
def source_reorder(request, notebook_pk):
    notebook = get_object_or_404(Notebook, pk=notebook_pk, owner=request.user)

    try:
        data = json.loads(request.body.decode("utf-8"))
        order = data.get("order", [])
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    for index, src_id in enumerate(order):
        try:
            src = notebook.sources.get(pk=src_id)
            src.position = index
            src.save(update_fields=["position"])
        except Source.DoesNotExist:
            continue

    return JsonResponse({"status": "ok"})


# =========================================================
# Export Notebook as TXT
# =========================================================

@login_required
def notebook_export(request, pk):
    notebook = get_object_or_404(Notebook, pk=pk, owner=request.user)

    lines = []
    lines.append(f"# {notebook.title}")
    if notebook.description:
        lines.append("")
        lines.append(notebook.description)

    lines.append("")
    lines.append("## Sources")

    for src in notebook.sources.all():
        lines.append("")
        lines.append(f"### {src.title} ({src.get_source_type_display()})")
        if src.content:
            lines.append(src.content)

    text = "\n".join(lines)
    response = HttpResponse(text, content_type="text/plain; charset=utf-8")
    safe_title = notebook.title or "notebook"
    response["Content-Disposition"] = f'attachment; filename="{safe_title}.txt"'
    return response
