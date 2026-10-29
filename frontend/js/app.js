const token = localStorage.getItem("approval_token") || "";
const approver = localStorage.getItem("approver") || "Priya";
localStorage.setItem("approver", approver);

let STATE = {};
let SCENES = {};
let inflight = false;

function headers() {
  const h = { "Content-Type": "application/json", "X-Approver-Id": approver };
  if (token) h.Authorization = "Bearer " + token;
  return h;
}

async function api(path, opts) {
  const r = await fetch(path, { headers: headers(), ...opts });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function format(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
}

function who(role) {
  if (role === "user") return "You";
  if (role === "assistant") return "Maya";
  if (role === "system") return "Priya";
  if (role === "gate") return "Gate";
  return role;
}

function threadEl() {
  return document.getElementById("transcript");
}

function scrollPane(id) {
  const el = document.getElementById(id);
  el.scrollTop = el.scrollHeight;
}

function renderMessage(msg, extraClass) {
  const node = document.createElement("div");
  node.className = `msg ${msg.role}${extraClass ? " " + extraClass : ""}`;
  node.innerHTML = `<div class="who">${esc(who(msg.role))}</div><div>${format(msg.text)}</div>`;
  return node;
}

function showTyping() {
  hideTyping();
  const wrap = document.createElement("div");
  wrap.className = "typing";
  wrap.id = "typing";
  wrap.innerHTML = "<i></i><i></i><i></i>";
  threadEl().appendChild(wrap);
  scrollPane("transcript");
}

function hideTyping() {
  document.getElementById("typing")?.remove();
}

function paintThread(messages) {
  const root = threadEl();
  if (!messages.length) {
    root.innerHTML = `<p class="thread-empty">Maya is waiting. Play a scene or type as the customer.</p>`;
    return;
  }
  root.innerHTML = "";
  messages.forEach((msg) => root.appendChild(renderMessage(msg)));
  scrollPane("transcript");
}

function setBusy(busy, paused) {
  inflight = busy;
  const chip = document.getElementById("status-chip");
  const prompt = document.getElementById("prompt");
  const send = document.getElementById("send-btn");
  if (paused) {
    chip.textContent = "Paused for Priya";
    chip.className = "chip pause";
  } else if (busy) {
    chip.textContent = "Maya is working";
    chip.className = "chip wait";
  } else {
    chip.textContent = "Ready";
    chip.className = "chip";
  }
  prompt.disabled = busy || paused;
  send.disabled = busy || paused;
}

function checksHtml(row) {
  const checks = row.checks || [];
  if (!checks.length && !row.retried) return "";
  const bits = checks.map((c) => {
    const name = c.subject ? `${c.kind}[${c.subject}]` : c.kind;
    const p = (c.p_ok == null || Number.isNaN(Number(c.p_ok))) ? "" : ` ${Number(c.p_ok).toFixed(2)}`;
    return name + p;
  });
  if (row.retried) bits.push("retried");
  return bits.length ? `<div class="muted checks">${esc(bits.join(" · "))}</div>` : "";
}

function paint(state) {
  STATE = state || {};
  hideTyping();
  setBusy(false, !!STATE.paused);
  document.getElementById("store").textContent = STATE.store || "Hearth & Cable";
  document.getElementById("today").textContent = STATE.today ? `today ${STATE.today}` : "";

  SCENES = {};
  (STATE.scenes || []).forEach((s) => { SCENES[s.id] = s; });
  document.getElementById("scenes").innerHTML = (STATE.scenes || []).map((s) =>
    `<button type="button" title="${esc(s.note)}" data-scene="${esc(s.id)}">${esc(s.title)}</button>`
  ).join("");

  paintThread(STATE.transcript || []);

  const tape = document.getElementById("tape");
  const rows = (STATE.tape || []).slice().reverse();
  tape.innerHTML = rows.length ? rows.map((row) =>
    `<div class="row"><span class="tag ${esc(row.outcome)}">${esc(row.outcome)}</span>
     <strong>${esc(row.tool)}</strong> · ${esc(row.source)} · ${esc(row.total_ms)}ms
     <div class="muted">${esc(row.reason)}</div>${checksHtml(row)}</div>`
  ).join("") : `<p class="muted">Tool gates and handbook evidence (relevance / sufficiency) land here.</p>`;

  const pending = STATE.pending || [];
  document.getElementById("desk").innerHTML = pending.length
    ? pending.map(card).join("")
    : `<p class="muted">Nothing waiting. High-risk refunds and shipped address changes pause here.</p>`;

  const board = STATE.board || {};
  const orders = (board.orders || []).map((o) =>
    `<tr><td>${esc(o.order_id)}</td><td><span class="pill ${esc(o.status)}">${esc(o.status)}</span></td>
     <td>$${esc(o.total)}</td><td>left $${esc(o.refundable_remaining)}</td><td>${esc(o.items)}</td></tr>`
  ).join("");
  const refunds = (board.refunds || []).map((r) => `${r.order_id} $${r.amount}`).join(", ") || "none";
  const emails = (board.emails || []).map((e) => e.to).join(", ") || "empty";
  document.getElementById("board").innerHTML = `
    <table><thead><tr><th>Order</th><th>Status</th><th>Total</th><th>Refundable</th><th>Item</th></tr></thead>
    <tbody>${orders}</tbody></table>
    <p class="muted">Ledger: ${esc(refunds)}. Outbox: ${esc(emails)}.</p>`;
}

function card(approval) {
  return `<div class="row">
    <strong>${esc(approval.tool)}</strong> · ${esc(approval.risk)} · ${esc(approval.source)}
    <div class="muted">${esc(approval.reason)}</div>
    <pre>${esc(JSON.stringify(approval.arguments, null, 2))}</pre>
    <pre>${esc(approval.excerpt || "")}</pre>
    <div class="desk-actions">
      <button type="button" class="approve" data-decide="${esc(approval.id)}" data-action="approve">Approve</button>
      <button type="button" class="deny" data-decide="${esc(approval.id)}" data-action="deny">Deny</button>
    </div>
  </div>`;
}

function optimistic(role, text) {
  const empty = threadEl().querySelector(".thread-empty");
  if (empty) empty.remove();
  threadEl().appendChild(renderMessage({ role, text }, "pending"));
  scrollPane("transcript");
}

async function sendText(text) {
  const cleaned = text.trim();
  if (!cleaned || inflight) return;
  if (STATE.paused) return;
  const prompt = document.getElementById("prompt");
  prompt.value = "";
  setBusy(true, false);
  optimistic("user", cleaned);
  showTyping();
  try {
    paint(await api("/demo/chat", { method: "POST", body: JSON.stringify({ text: cleaned }) }));
  } catch (err) {
    hideTyping();
    optimistic("gate", String(err.message || err));
    setBusy(false, !!STATE.paused);
  }
}

async function play(scene) {
  if (!scene || inflight) return;
  if (STATE.paused && scene.kind !== "claim") return;
  if (scene.kind === "claim") {
    setBusy(true, !!STATE.paused);
    optimistic("assistant", scene.prompt);
    showTyping();
    try {
      paint(await api("/demo/claim", { method: "POST", body: JSON.stringify({ claim: scene.prompt }) }));
    } catch (err) {
      hideTyping();
      optimistic("gate", String(err.message || err));
      setBusy(false, !!STATE.paused);
    }
    return;
  }
  await sendText(scene.prompt);
}

document.getElementById("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  sendText(document.getElementById("prompt").value);
});

document.getElementById("scenes").addEventListener("click", (event) => {
  const btn = event.target.closest("[data-scene]");
  if (!btn) return;
  play(SCENES[btn.dataset.scene]);
});

document.getElementById("desk").addEventListener("click", async (event) => {
  const btn = event.target.closest("[data-decide]");
  if (!btn || inflight) return;
  setBusy(true, false);
  showTyping();
  try {
    paint(await api(`/demo/${btn.dataset.action}/${btn.dataset.decide}`, {
      method: "POST",
      body: JSON.stringify({ approver, note: "" }),
    }));
  } catch (err) {
    hideTyping();
    optimistic("gate", String(err.message || err));
    setBusy(false, !!STATE.paused);
  }
});

document.getElementById("reset-btn").addEventListener("click", async () => {
  if (!confirm("Reset orders, chat, tape and pending approvals?")) return;
  paint(await api("/demo/reset", { method: "POST" }));
});

api("/demo/state").then(paint).catch((err) => {
  threadEl().textContent = err.message;
});
