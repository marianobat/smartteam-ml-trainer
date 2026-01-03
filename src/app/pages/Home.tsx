import { useEffect, useMemo, useState } from "react";
import { API_BASE, EXT_URL, TEMPLATE_SB3, TW_EDITOR } from "../../core/bridge/config";
import { getRoom, getToken, setRoom as setSessionRoom, setToken } from "../../core/bridge/session";
import "./Home.css";

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
  const extensionUrl = EXT_URL.trim();
  const templateUrl = TEMPLATE_SB3.trim();

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
    <div className="home">
      <header className="home-header">
        <div>
          <div className="home-kicker">SmartTEAM</div>
          <h1 className="home-title">SmartTEAM IA</h1>
          <p className="home-subtitle">
            Crea una sesión para compartir el room entre el entrenador y TurboWarp.
          </p>
        </div>
        <div className="home-session">
          <div className="home-session-row">
            <button onClick={handleCreateSession} disabled={isCreating}>
              {isCreating ? "Creando..." : "Crear sesión"}
            </button>
            <div className="home-session-status">
              Estado: <b>{status === "ready" ? "lista" : status === "error" ? "error" : "inactiva"}</b>
            </div>
          </div>
          <div className="home-room">
            <div className="home-room-label">Room</div>
            <div className="home-room-row">
              <div className="home-room-code">{room || "—"}</div>
              <button onClick={handleCopy} disabled={!room}>
                Copiar
              </button>
              {copyNotice && <div className="home-room-note">{copyNotice}</div>}
            </div>
          </div>
          {error && <div className="home-session-error">{error}</div>}
          <div className="home-session-note">
            Si refrescás la página, la sesión se mantiene solo en esta pestaña.
          </div>
        </div>
      </header>

      <section className="home-panels">
        <article className="home-panel home-panel--trainer">
          <div className="home-panel-media">
            <span>Imagen</span>
          </div>
          <div className="home-panel-body">
            <h2 className="home-panel-title">Entrenador</h2>
            <p className="home-panel-copy">
              Entrená modelos en tiempo real y publicá los gestos al bridge.
            </p>
            <div className="home-panel-actions">
              <button
                onClick={() => window.location.assign(`${baseUrl}trainer?room=${encodeURIComponent(room)}`)}
                disabled={!canEnter}
              >
                Abrir entrenador
              </button>
            </div>
          </div>
        </article>

        <article className="home-panel home-panel--program">
          <div className="home-panel-media">
            <span>Imagen</span>
          </div>
          <div className="home-panel-body">
            <h2 className="home-panel-title">Programador</h2>
            <p className="home-panel-copy">
              Abrí TurboWarp listo para recibir el room y la extensión.
            </p>
            <div className="home-panel-actions">
              <button
                onClick={() => {
                  if (!room) return;
                  const params = new URLSearchParams();
                  params.set("room", room);
                  if (templateUrl) {
                    params.set("project_url", templateUrl);
                  }
                  if (extensionUrl) {
                    params.set("extension", extensionUrl);
                  }
                  window.location.assign(`${TW_EDITOR}?${params.toString()}`);
                }}
                disabled={!canEnter}
              >
                Abrir TurboWarp
              </button>
            </div>
          </div>
        </article>
      </section>
    </div>
  );
}
