import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "../../core/bridge/config";
import { getRoom, getToken, setRoom as setSessionRoom, setToken } from "../../core/bridge/session";

type SessionResponse = {
  room: string;
  publishToken: string;
  wsBaseUrl: string;
  expiresInSec: number;
};

export default function Home() {
  const [room, setRoom] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [status, setStatus] = useState<"idle" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [copyNotice, setCopyNotice] = useState<string | null>(null);

  const baseUrl = useMemo(() => import.meta.env.BASE_URL ?? "/", []);
  const canEnter = Boolean(room);

  useEffect(() => {
    const storedRoom = getRoom();
    const storedToken = getToken();
    if (storedRoom && storedToken) {
      setRoom(storedRoom);
      setStatus("ready");
    }
  }, []);

  const handleCreateSession = async () => {
    setIsCreating(true);
    setError(null);
    setCopyNotice(null);
    try {
      const response = await fetch(`${API_BASE}/session/new`, { method: "POST" });
      if (!response.ok) {
        throw new Error(`Error ${response.status}`);
      }
      const data = (await response.json()) as SessionResponse;
      setRoom(data.room);
      setSessionRoom(data.room);
      setToken(data.publishToken);
      setStatus("ready");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setError("No se pudo crear la sesión. Probá de nuevo.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!room) return;
    try {
      await navigator.clipboard.writeText(room);
      setCopyNotice("Copiado ✅");
    } catch (err) {
      console.error(err);
      setCopyNotice("No se pudo copiar.");
    }
  };

  return (
    <div style={{ padding: 16, maxWidth: 640, margin: "0 auto", display: "grid", gap: 12 }}>
      <div>
        <h1 style={{ marginBottom: 4 }}>SmartTEAM Lobby</h1>
        <div style={{ fontSize: 13, opacity: 0.8 }}>
          Crea una sesión para compartir el room entre el entrenador y TurboWarp.
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={handleCreateSession} disabled={isCreating}>
          {isCreating ? "Creando..." : "Crear sesión"}
        </button>
        <div style={{ fontSize: 12, opacity: 0.8 }}>
          Estado: <b>{status === "ready" ? "lista" : status === "error" ? "error" : "inactiva"}</b>
        </div>
      </div>

      <div
        style={{
          border: "1px solid #ddd",
          borderRadius: 12,
          padding: 12,
          display: "grid",
          gap: 8,
          background: "#fafafa",
        }}
      >
        <div style={{ fontSize: 12, opacity: 0.8 }}>Room</div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 14 }}>
            {room || "—"}
          </div>
          <button onClick={handleCopy} disabled={!room}>
            Copiar
          </button>
          {copyNotice && <div style={{ fontSize: 12, opacity: 0.8 }}>{copyNotice}</div>}
        </div>
        {error && <div style={{ color: "#b91c1c", fontSize: 12 }}>{error}</div>}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={() => window.location.assign(`${baseUrl}trainer?room=${encodeURIComponent(room)}`)}
          disabled={!canEnter}
        >
          Entrenador
        </button>
        <button
          onClick={() => window.location.assign(`${baseUrl}program?room=${encodeURIComponent(room)}`)}
          disabled={!canEnter}
        >
          Programador
        </button>
      </div>

      <div style={{ fontSize: 12, opacity: 0.7 }}>
        Si refrescás la página, la sesión se mantiene solo en esta pestaña.
      </div>
    </div>
  );
}
