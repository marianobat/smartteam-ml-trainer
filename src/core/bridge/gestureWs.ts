export type WsStatus = "idle" | "connecting" | "open" | "reconnecting" | "error";
export type WsRole = "publisher" | "subscriber";

export type GestureMessage = {
  type: "gesture";
  label: string;
  confidence: number;
  seq?: number;
  ts?: number;
};

type HelloMessage = {
  type: "hello";
  room: string;
  role: WsRole;
};

type PresenceMessage = {
  type: "presence";
  subscribers: number;
};

type ErrorMessage = {
  type: "error";
  message: string;
};

type GestureWsHandlers = {
  onStatus?: (status: WsStatus) => void;
  onHello?: (message: HelloMessage) => void;
  onPresence?: (subscribers: number) => void;
  onError?: (message: string) => void;
};

let socket: WebSocket | null = null;
let reconnectTimer: number | null = null;
let reconnectAttempt = 0;
let currentUrl = "";
let allowReconnect = false;
let handlers: GestureWsHandlers = {};

const clearReconnectTimer = () => {
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
};

const notifyStatus = (status: WsStatus) => {
  handlers.onStatus?.(status);
};

const reportError = (message: string) => {
  console.error("Gesture WS:", message);
  handlers.onError?.(message);
};

const BACKOFF_STEPS = [1000, 2000, 3000, 5000];

const getBackoffDelay = (attempt: number) => {
  const index = Math.min(Math.max(attempt - 1, 0), BACKOFF_STEPS.length - 1);
  return BACKOFF_STEPS[index];
};

const scheduleReconnect = () => {
  if (reconnectTimer !== null || !allowReconnect || !currentUrl) return;
  const delay = getBackoffDelay(reconnectAttempt || 1);
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    if (!allowReconnect || !currentUrl) return;
    openSocket();
  }, delay);
  notifyStatus("reconnecting");
};

const openSocket = () => {
  clearReconnectTimer();
  notifyStatus(reconnectAttempt > 0 ? "reconnecting" : "connecting");
  try {
    socket = new WebSocket(currentUrl);
  } catch {
    reportError("No se pudo abrir el WebSocket.");
    reconnectAttempt += 1;
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    reconnectAttempt = 0;
    notifyStatus("open");
  };

  socket.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    try {
      const payload = JSON.parse(event.data) as HelloMessage | PresenceMessage | ErrorMessage;
      if (payload.type === "hello") {
        handlers.onHello?.(payload);
      } else if (payload.type === "presence") {
        handlers.onPresence?.(payload.subscribers);
      } else if (payload.type === "error") {
        reportError(payload.message);
      }
    } catch (err) {
      console.error("Gesture WS parse error:", err);
    }
  };

  socket.onclose = () => {
    socket = null;
    if (allowReconnect) {
      reconnectAttempt += 1;
      scheduleReconnect();
    } else {
      notifyStatus("idle");
    }
  };

  socket.onerror = () => {
    reportError("Error en WebSocket.");
  };
};

export function connectGestureWs(url: string, nextHandlers: GestureWsHandlers = {}): void {
  handlers = nextHandlers;
  if (!url) {
    notifyStatus("error");
    reportError("Falta URL de WebSocket.");
    return;
  }
  currentUrl = url;
  allowReconnect = true;

  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  openSocket();
}

export function sendGesture(message: GestureMessage): void {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  try {
    socket.send(JSON.stringify(message));
  } catch {
    reportError("No se pudo enviar el gesto.");
  }
}

export function disconnectGestureWs(): void {
  allowReconnect = false;
  currentUrl = "";
  clearReconnectTimer();
  reconnectAttempt = 0;
  handlers = {};

  if (socket) {
    try {
      socket.close();
    } catch (err) {
      console.error("Gesture WS close error:", err);
    }
    socket = null;
  }
  notifyStatus("idle");
}
