import { useEffect, useMemo, useState } from "react";
import {
  Hand,
  Smile,
  PersonStanding,
  Image as ImageIcon,
  Pencil,
  Mic,
  Sparkles,
  Satellite,
  Check,
} from "lucide-react";
import { API_BASE, EXT_URL, TEMPLATE_SB3, TW_EDITOR } from "../../core/bridge/config";
import { TURBOWARP_ENABLED } from "../../core/bridge/features";
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

  const handleOpenTrainer = () => {
    // La sesión (room) solo hace falta para publicar a TurboWarp; el
    // entrenador y el micro:bit funcionan sin ella.
    const suffix = room ? `?room=${encodeURIComponent(room)}` : "";
    window.location.assign(`${baseUrl}trainer${suffix}`);
  };

  const handleOpenProgram = () => {
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
  };

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
      setError("No se pudo crear la sesión. Prueba de nuevo.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!room) return;
    try {
      await navigator.clipboard.writeText(room);
      setCopyNotice("¡Copiado!");
    } catch (err) {
      console.error(err);
      setCopyNotice("No se pudo copiar.");
    }
  };

  return (
    <div className="home">
      <header className="home-hero">
        <div className="home-kicker">SmartTEAM</div>
        <h1 className="home-title">SmartTEAM IA</h1>
        <p className="home-subtitle">
          Enséñale a la computadora con tus manos, tu cuerpo, tu voz o tus dibujos — y conecta lo
          que aprende a un micro:bit
          {TURBOWARP_ENABLED ? " o a TurboWarp" : ""}.
        </p>
      </header>

      <main className="home-cards">
        <article className="home-card home-card--trainer">
          <div className="home-card-icon" aria-hidden="true">
            <Hand size={26} />
            <Smile size={26} />
            <PersonStanding size={26} />
            <ImageIcon size={26} />
            <Pencil size={26} />
            <Mic size={26} />
          </div>
          <h2 className="home-card-title">Entrenador</h2>
          <p className="home-card-copy">
            Elige una modalidad, enséñale con ejemplos, entrena tu modelo y pruébalo en vivo.
          </p>
          <button type="button" className="home-card-cta" onClick={handleOpenTrainer}>
            <Sparkles size={18} aria-hidden="true" /> Abrir entrenador
          </button>
        </article>

        {TURBOWARP_ENABLED && (
        <article className="home-card home-card--program">
          <div className="home-card-icon" aria-hidden="true">
            <Satellite size={26} />
          </div>
          <h2 className="home-card-title">TurboWarp</h2>
          <p className="home-card-copy">
            Para programar en Scratch con lo que detecta tu modelo, primero crea una sesión y
            comparte el room.
          </p>

          <div className="home-session">
            <div className="home-session-row">
              <button type="button" onClick={() => void handleCreateSession()} disabled={isCreating}>
                {isCreating ? "Creando..." : "Crear sesión"}
              </button>
              <span className="home-session-status">
                {status === "ready" ? (
                  <>
                    <Check size={14} aria-hidden="true" /> lista
                  </>
                ) : status === "error" ? (
                  "error"
                ) : (
                  "sin sesión"
                )}
              </span>
            </div>
            <div className="home-room">
              <span className="home-room-code">{room || "—"}</span>
              <button type="button" onClick={() => void handleCopy()} disabled={!room}>
                Copiar
              </button>
              {copyNotice && <span className="home-room-note">{copyNotice}</span>}
            </div>
            {error && <div className="home-session-error">{error}</div>}
          </div>

          <button
            type="button"
            className="home-card-cta home-card-cta--secondary"
            onClick={handleOpenProgram}
            disabled={!canEnter}
          >
            Abrir TurboWarp
          </button>
          <p className="home-card-note">
            La sesión solo hace falta para TurboWarp: el entrenador y el micro:bit funcionan sin
            ella.
          </p>
        </article>
        )}
      </main>
    </div>
  );
}
