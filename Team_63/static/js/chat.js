document.getElementById("chat-form").addEventListener("submit", function (e) {
  e.preventDefault();

  const input = document.getElementById("user-input");
  const msg = input.value.trim();
  if (msg === "") return;

  const box = document.getElementById("chat-box");

  // Sanitize user input to prevent XSS
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Show user message instantly (escaped)
  const userMsgDiv = document.createElement("div");
  userMsgDiv.className = "mb-3";
  userMsgDiv.innerHTML = `<div class="user-msg"><b>You:</b> ${escapeHtml(msg)}</div>`;
  box.appendChild(userMsgDiv);

  // Create and show loader INSIDE the chat box (so it's visible)
  const loaderDiv = document.createElement("div");
  loaderDiv.id = "loading";
  loaderDiv.className = "show";
  loaderDiv.innerHTML = `
    <img src="https://i.gifer.com/ZZ5H.gif" width="60" alt="Loading..." />
    <p class="text-muted fw-semibold">Generating response...</p>
  `;
  box.appendChild(loaderDiv);

  // Auto scroll to show loader
  box.scrollTop = box.scrollHeight;

  input.value = "";
  input.disabled = true;

  // Get CSRF token
  const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

  if (!csrftoken) {
    console.warn("CSRF token not found - make sure {% csrf_token %} is in your form");
  }

  fetch("/rag-chat-api/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(csrftoken && { "X-CSRFToken": csrftoken }),
    },
    body: JSON.stringify({ query: msg }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      return res.json();
    })
    .then((data) => {
      // Remove loader
      loaderDiv.remove();
      input.disabled = false;
      input.focus();

      if (data.answer) {
        const aiDiv = document.createElement("div");
        aiDiv.className = "mb-3";
        aiDiv.innerHTML = `
          <div class="ai-title"><b>AI:</b></div>
          <div class="flowchart-container ai-response" data-raw="${escapeHtml(data.answer)}"></div>
        `;
        box.appendChild(aiDiv);
      } else if (data.error) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "mb-3";
        errorDiv.innerHTML = `<p class="text-danger"><b>Error:</b> ${escapeHtml(data.error)}</p>`;
        box.appendChild(errorDiv);
      }

      // Auto scroll
      box.scrollTop = box.scrollHeight;
    })
    .catch((err) => {
      // Remove loader
      loaderDiv.remove();
      input.disabled = false;
      input.focus();

      const errorDiv = document.createElement("div");
      errorDiv.className = "mb-3";
      errorDiv.innerHTML = `<p class="text-danger"><b>Error:</b> Network issue - ${escapeHtml(err.message)}</p>`;
      box.appendChild(errorDiv);

      console.error("Chat API Error:", err);
      box.scrollTop = box.scrollHeight;
    });
});