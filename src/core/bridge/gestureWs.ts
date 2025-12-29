let socket: WebSocket | null = null;
let reconnectTimer: number | null = null;
let lastSentLabel = "";
let currentUrl = "";
let allowReconnect = true;

const RECONNECT_DELAY_MS = 1000;

const clearReconnectTimer = () => {
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
};

const scheduleReconnect = () => {
  if (reconnectTimer !== null || !allowReconnect || !currentUrl) return;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    if (!allowReconnect || !currentUrl) return;
    connectGestureWs(currentUrl);
  }, RECONNECT_DELAY_MS);
};

export function connectGestureWs(url: string): void {
  if (!url) return;
  currentUrl = url;
  allowReconnect = true;

  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  clearReconnectTimer();
  lastSentLabel = "";

  try {
    socket = new WebSocket(url);
  } catch (err) {
    console.error("Gesture WS connect error:", err);
    scheduleReconnect();
    return;
  }

  socket.onclose = () => {
    socket = null;
    if (allowReconnect) {
      scheduleReconnect();
    }
  };

  socket.onerror = (err) => {
    console.error("Gesture WS error:", err);
  };
}

export function sendStableLabel(label: string): void {
  if (!label || label === lastSentLabel) return;
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  try {
    socket.send(label);
    lastSentLabel = label;
  } catch (err) {
    console.error("Gesture WS send error:", err);
  }
}

export function disconnectGestureWs(): void {
  allowReconnect = false;
  currentUrl = "";
  clearReconnectTimer();
  lastSentLabel = "";

  if (socket) {
    try {
      socket.close();
    } catch (err) {
      console.error("Gesture WS close error:", err);
    }
    socket = null;
  }
}
