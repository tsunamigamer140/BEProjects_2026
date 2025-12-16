// ai_notebook/static/ai_notebook/notebook.js
document.addEventListener("DOMContentLoaded", function () {
  const chatWindow = document.getElementById("chat-window");
  if (chatWindow) {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
});
