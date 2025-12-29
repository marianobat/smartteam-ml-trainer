const DEFAULT_WS_URL = "wss://smartteam-gesture-bridge.marianobat.workers.dev/ws?room=demo";

export function getWsUrl(): string {
  const params =
    typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
  const queryUrl = params?.get("ws") ?? "";
  if (queryUrl.trim()) {
    return queryUrl.trim();
  }

  const envUrl = import.meta.env.VITE_GESTURE_WS_URL as string | undefined;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim();
  }

  return DEFAULT_WS_URL;
}
