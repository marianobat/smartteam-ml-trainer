const ROOM_KEY = "smartteam.session.room";
const TOKEN_KEY = "smartteam.session.publishToken";

export function getRoom(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(ROOM_KEY);
}

export function setRoom(room: string): void {
  if (typeof window === "undefined") return;
  const value = room.trim();
  if (!value) {
    window.sessionStorage.removeItem(ROOM_KEY);
    return;
  }
  window.sessionStorage.setItem(ROOM_KEY, value);
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  const value = token.trim();
  if (!value) {
    window.sessionStorage.removeItem(TOKEN_KEY);
    return;
  }
  window.sessionStorage.setItem(TOKEN_KEY, value);
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(ROOM_KEY);
  window.sessionStorage.removeItem(TOKEN_KEY);
}
