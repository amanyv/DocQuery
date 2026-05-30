function log(type, message, data = null) {
  const time = new Date().toISOString();
  console[type](`[${time}] ${message}`, data || "");
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function smoothScrollToBottom(container, force = false) {
  if (!container) return;
  
  const isNearBottom = 
    container.scrollHeight - container.scrollTop - container.clientHeight < 150;
  
  if (isNearBottom || force) {
    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'smooth'
    });
  }
}

function scrollToBottom() {
  const chat = document.getElementById("chat");
  chat.scrollTo({
    top: chat.scrollHeight,
    behavior: 'smooth'
  });
  updateScrollHint();
}

function updateScrollHint() {
  const chat = document.getElementById("chat");
  const hint = document.getElementById("scroll-hint");
  
  if (!chat || !hint) return;
  
  const isAtBottom = 
    chat.scrollHeight - chat.scrollTop - chat.clientHeight < 50;
  
  if (isAtBottom) {
    hint.classList.remove('visible');
  } else {
    hint.classList.add('visible');
  }
}

function showError(message, container) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'msg bot error';
  errorDiv.innerHTML = `<strong>⚠️ Error:</strong> ${escapeHtml(message)}`;
  container.appendChild(errorDiv);
  smoothScrollToBottom(container, true);
}


// const API = "http://127.0.0.1:5000";
const API = "https://docquery-5dai.onrender.com";

const MAX_CHARS = 100;
let pollInterval = null;
let isLoading = false;
let chatHistory = [];
let rafId = null;
let pendingUpdate = false;

let userId = localStorage.getItem("docquery_user_id");
if (!userId) {
    userId = crypto.randomUUID(); 
    localStorage.setItem("docquery_user_id", userId);
}
console.log("Active User ID:", userId);

const input = document.getElementById("q");
const counter = document.getElementById("word-counter");
const chat = document.getElementById("chat");

const updateCounter = debounce(() => {
  const chars = input.value.length;
  const percentage = (chars / MAX_CHARS) * 100;

  counter.textContent = `${chars} / ${MAX_CHARS} characters`;

  counter.classList.remove('warning', 'error');
  input.classList.remove('warning', 'error-border');

  if (chars > MAX_CHARS) {
    counter.classList.add('error');
    input.classList.add('error-border');
  } else if (percentage > 80) {
    counter.classList.add('warning');
    input.classList.add('warning');
  }
}, 100);

input.addEventListener("input", updateCounter);
chat.addEventListener('scroll', debounce(updateScrollHint, 100));

window.addEventListener('DOMContentLoaded', () => {
  console.log("[DocQuery] Running window load session reset...");
  fetch(API + "/api/reset", { 
    method: 'POST',
    headers: { "X-User-ID": userId }
  })
    .then(res => res.json())
    .then(data => {
      console.log("[DocQuery] Backend database cleared successfully:", data);
      localStorage.removeItem("fileSizes");
      loadFiles();
    })
    .catch(err => {
      console.error("[DocQuery] Failed to trigger auto-reset endpoint:", err);
      loadFiles();
    });
});

async function loadFiles() {
  try {
    const res = await fetch(API + "/api/files", {
      headers: { "X-User-ID": userId }
    });
    
    const data = await res.json();
    renderFiles(data.files || []);
  } catch (e) {
    log("error", "Failed to load files", e);
  }
}

function renderFiles(files) {
  const list = document.getElementById("file-list");
  list.innerHTML = "";
  if (files.length === 0) {
    list.innerHTML =
      '<div style="color:#888;font-size:12px;">No PDFs uploaded yet</div>';
    return;
  }
  files.forEach((f) => {
    const item = document.createElement("div");
    item.className = "file-item";
    const span = document.createElement("span");
    span.title = f;
    span.style =
      "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const stored = JSON.parse(localStorage.getItem("fileSizes") || "{}");
    const size = stored[f];

    let sizeText = "";
    if (size) {
      sizeText = ` (${(size / (1024 * 1024)).toFixed(2)} MB)`;
    }

    span.textContent = "📄 " + f + sizeText;
    const btn = document.createElement("button");
    btn.className = "del-btn";
    btn.textContent = "✕";
    btn.addEventListener("click", () => deleteFile(f));
    item.appendChild(span);
    item.appendChild(btn);
    list.appendChild(item);
  });
}

async function uploadFiles(input) {
  const status = document.getElementById("upload-status");
  const uploadBtn = document.getElementById("upload-btn");
  const files = input.files;
  if (!files.length) return;

  const MAX_SIZE = 25 * 1024 * 1024;

  for (const file of files) {
    if (file.size > MAX_SIZE) {
      status.textContent = `❌ ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds 25MB`;
      status.style.color = "#f87171";
      setTimeout(() => {
        status.textContent = "";
        status.style.color = "#a5f3a0";
      }, 4000);
      input.value = "";
      return;
    }
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  
  const namesWithSize = Array.from(files)
    .map((f) => `${f.name} (${(f.size / (1024 * 1024)).toFixed(2)} MB)`)
    .join(", ");

  status.textContent = `⏳ Uploading: ${namesWithSize}`;
  status.style.color = "#c7d2fe";
  log("info", "Uploading files", namesWithSize);

  try {
    const formData = new FormData();
    for (const file of files) formData.append("files", file);
    
    const res = await fetch(API + "/api/upload", {
      method: "POST",
      headers: {
        "X-User-ID": userId
      },
      body: formData,
    });
    const data = await res.json();
    console.log("[upload] server response:", data);

    if (data.error) {
      console.error("[upload] error:", data.error);
      status.textContent = "❌ " + data.error;
      status.style.color = "#f87171";
    } else {
      const stored = JSON.parse(localStorage.getItem("fileSizes") || "{}");

      for (const file of files) {
        stored[file.name] = file.size;
      }

      localStorage.setItem("fileSizes", JSON.stringify(stored));

      console.log("[upload] success, files:", data.files);
      status.textContent = "✅ Uploaded successfully";
      status.style.color = "#a5f3a0";
      setTimeout(() => status.textContent = "", 3000);
      loadFiles();
      startPolling();
    }
  } catch (e) {
    log("error", "Upload failed", e);
    status.textContent = "❌ Upload failed. Please try again.";
    status.style.color = "#f87171";
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "+ Upload PDF";
    input.value = "";
  }
}

async function deleteFile(filename) {
  if (!confirm(`Delete ${filename}?`)) return;
  log("info", "Delete file", filename);
  
  try {
    const res = await fetch(
      API + `/api/files/${encodeURIComponent(filename)}`,
      { method: "DELETE" ,
      headers: { "X-User-ID": userId }
    },
    );
    const data = await res.json();
    console.log("[delete] response:", data);
    
    if (!data.error) {
      const stored = JSON.parse(localStorage.getItem("fileSizes") || "{}");
      delete stored[filename];
      localStorage.setItem("fileSizes", JSON.stringify(stored));
      
      loadFiles();
    } else {
      alert("Failed to delete file: " + data.error);
    }
  } catch (e) {
    log("error", "Delete failed", e);
    alert("Failed to delete file. Please try again.");
  }
}

function startPolling() {
  const banner = document.getElementById("indexing-banner");
  banner.style.display = "flex";

  if (pollInterval) return;
  
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(API + "/api/status");
      const data = await res.json();

      console.log("[polling] status:", data);

      const ready = data.ready && !data.indexing;
      document.getElementById("send-btn").disabled = !ready;

      document.querySelectorAll(".qa-btn").forEach((btn) => {
        btn.disabled = !ready;
      });

      if (ready) {
        clearInterval(pollInterval);
        pollInterval = null;
        banner.style.display = "none";
        console.log("[polling] stopped - ready");
      }
    } catch (e) {
      console.error("[polling] error:", e);
    }
  }, 5000);
}

function setLoading(val) {
  isLoading = val;
  document.getElementById("send-btn").disabled = val;
  document.getElementById("q").disabled = val;
  document.querySelectorAll(".qa-btn").forEach((b) => (b.disabled = val));
}

function scheduleRender(element, rawText, container) {
  if (!pendingUpdate) {
    pendingUpdate = true;
    rafId = requestAnimationFrame(() => {
      element.innerHTML = marked.parse(rawText);
      smoothScrollToBottom(container);
      pendingUpdate = false;
    });
  }
}

function processLatexEscapes(text) {
  return text
    .replace(/\\\[/g, "$$")
    .replace(/\\\]/g, "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

async function send() {
  if (isLoading) return;
  const q = document.getElementById("q").value.trim();

  if (q.length > MAX_CHARS) {
    alert(`❌ Query too long. Max ${MAX_CHARS} characters allowed.`);
    return;
  }

  if (!q) return;
  log("info", "ASK sent", { question: q });

  document.getElementById("q").value = "";
  updateCounter();
  setLoading(true);

  const userDiv = document.createElement("div");
  userDiv.className = "msg user";
  userDiv.textContent = q;
  chatHistory.push({
    role: "user",
    content: q,
  });
  chat.appendChild(userDiv);
  smoothScrollToBottom(chat, true);

  const card = document.createElement("div");
  card.className = "qa-card";

  const aDiv = document.createElement("div");
  aDiv.className = "qa-answer streaming-cursor";

  card.appendChild(aDiv);
  chat.appendChild(card);
  smoothScrollToBottom(chat, true);

  let rawText = "";
  let sources = [];

  try {
    log("info", "Streaming started");
    const res = await fetch(API + "/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json",
        "X-User-ID": userId
       },
      body: JSON.stringify({ question: q, history: chatHistory }),
    });

    if (!res.ok) {
      const err = await res.json();
      aDiv.classList.remove("streaming-cursor");
      aDiv.innerHTML = marked.parse(err.error || "Something went wrong.");
      setLoading(false);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;

        try {
          const json = JSON.parse(payload);
          if (json.error) {
            console.warn("[ask] stream error:", json.error);
            aDiv.classList.remove("streaming-cursor");
            aDiv.innerHTML = marked.parse(json.error);
          } else if (json.sources) {
            console.log("[ask] sources:", json.sources);
            sources = json.sources;
          } else if (json.token) {
            log("debug", "Token", json.token);
            rawText += json.token;
            rawText = processLatexEscapes(rawText);

            scheduleRender(aDiv, rawText, chat);
          }
        } catch (err) {
          console.error(
            "[ask] SSE parse error:",
            err,
            "raw payload:",
            payload,
          );
        }
      }
    }
  } catch (e) {
    log("error", "Network error", e);
    aDiv.classList.remove("streaming-cursor");
    aDiv.innerHTML = marked.parse("❌ Network error. Please try again.");
  }

  aDiv.classList.remove("streaming-cursor");
  rawText = processLatexEscapes(rawText);
  aDiv.innerHTML = marked.parse(rawText || "No response received.");

  if (sources.length > 0) {
    const row = document.createElement("div");
    row.className = "sources-row";
    sources.forEach((s) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = `📄 ${s.file} p.${s.page}`;

      chip.onclick = () => openPdf(s.file, s.page);
      
      row.appendChild(chip);
    });
    card.appendChild(row);
  }

  smoothScrollToBottom(chat, true);

  chatHistory.push({
    role: "assistant",
    content: rawText,
  });

  setLoading(false);
}

async function runSummarize() {
  if (isLoading) return;
  setLoading(true);

  const botDiv = document.createElement("div");
  botDiv.className = "msg bot streaming-cursor";
  chat.appendChild(botDiv);
  smoothScrollToBottom(chat, true);

  let rawText = "";

  try {
    console.log("[summarize] request sent");
    const res = await fetch(API + "/api/summarize", { method: "POST", headers: { "X-User-ID": userId } });

    if (!res.ok) {
      const err = await res.json();
      botDiv.classList.remove("streaming-cursor");
      botDiv.innerHTML = marked.parse(
        err.error || "Something went wrong.",
      );
      setLoading(false);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;

        try {
          const json = JSON.parse(payload);
          if (json.token) {
            rawText += json.token;
            rawText = processLatexEscapes(rawText);
            
            scheduleRender(botDiv, rawText, chat);
          } else if (json.error) {
            console.warn("[summarize] stream error:", json.error);
            botDiv.innerHTML = marked.parse(json.error);
          }
        } catch (_) {}
      }
    }
  } catch (e) {
    console.error("[summarize] network error:", e);
    botDiv.innerHTML = marked.parse("❌ Network error. Please try again.");
  }

  botDiv.classList.remove("streaming-cursor");
  rawText = processLatexEscapes(rawText);
  botDiv.innerHTML = marked.parse(rawText || "No summary available.");

  smoothScrollToBottom(chat, true);
  setLoading(false);
}

function clearContext() {
  if (chat.children.length === 0) return;
  
  if (confirm("Clear all chat messages?")) {
    chat.innerHTML = "";
    chatHistory = [];
    log("info", "Chat cleared");
  }
}

console.log("DocQuery initialized");
updateScrollHint();

function openPdf(filename, page) {
  const pane = document.getElementById("pdf-pane");
  const frame = document.getElementById("pdf-frame");
  const title = document.getElementById("pdf-title");

  title.textContent = `${filename} (Page ${page})`;

  frame.src = `${API}/api/files/${encodeURIComponent(filename)}?user=${userId}#page=${page}`;

  pane.classList.add("open");
}

function closePdf() {
  const pane = document.getElementById("pdf-pane");
  pane.classList.remove("open");

  setTimeout(() => {
    document.getElementById("pdf-frame").src = "";
  }, 300);
}