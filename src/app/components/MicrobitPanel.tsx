// src/app/components/MicrobitPanel.tsx
//
// Panel "Conectar micro:bit" compartido por todos los entrenadores. Recibe la
// etiqueta estable y su confianza; aplica el umbral y publica por Web Serial
// con la misma semántica que el publicador WebSocket: envía al cambiar la
// etiqueta y como heartbeat cada RESEND_INTERVAL_MS.

import { useEffect, useRef, useState } from "react";
import {
  connectMicrobit,
  disconnectMicrobit,
  isWebSerialSupported,
  sendMicrobitLabel,
} from "../../core/microbit/serialConnection";
import {
  DEFAULT_CONFIDENCE_THRESHOLD,
  NONE_LABEL,
  RESEND_INTERVAL_MS,
} from "../../core/microbit/protocol";

const MAX_LOG_LINES = 6;

type PanelStatus = "idle" | "connecting" | "open" | "error";

type MicrobitPanelProps = {
  /** Etiqueta estable actual ("" o NONE_LABEL si no hay detección). */
  label: string;
  confidence: number;
};

export default function MicrobitPanel({ label, confidence }: MicrobitPanelProps) {
  const supported = isWebSerialSupported();
  const [status, setStatus] = useState<PanelStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [log, setLog] = useState<string[]>([]);

  const statusRef = useRef<PanelStatus>(status);
  statusRef.current = status;

  const labelToSendRef = useRef(NONE_LABEL);
  labelToSendRef.current =
    label && label !== NONE_LABEL && confidence >= threshold ? label : NONE_LABEL;

  const lastSentRef = useRef<string>("");

  const pushLog = (line: string) => {
    setLog((prev) => [line, ...prev].slice(0, MAX_LOG_LINES));
  };

  const trySend = async (force: boolean) => {
    if (statusRef.current !== "open") return;
    const labelToSend = labelToSendRef.current;
    if (!force && labelToSend === lastSentRef.current) return;
    try {
      const line = await sendMicrobitLabel(labelToSend);
      lastSentRef.current = labelToSend;
      pushLog(line);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // Envío inmediato cuando cambia la etiqueta efectiva
  useEffect(() => {
    void trySend(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, confidence, threshold, status]);

  // Heartbeat: reenvía la etiqueta actual cada RESEND_INTERVAL_MS
  useEffect(() => {
    if (status !== "open") return;
    const id = window.setInterval(() => {
      void trySend(true);
    }, RESEND_INTERVAL_MS);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Desconexión limpia al desmontar
  useEffect(() => {
    return () => {
      void disconnectMicrobit();
    };
  }, []);

  const handleConnect = async () => {
    setError(null);
    setStatus("connecting");
    try {
      await connectMicrobit();
      lastSentRef.current = "";
      setStatus("open");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // El usuario canceló el diálogo de puertos: no es un error
      if (/no port selected/i.test(message)) {
        setStatus("idle");
        return;
      }
      setStatus("error");
      setError(message);
    }
  };

  const handleDisconnect = async () => {
    await disconnectMicrobit();
    setStatus("idle");
    setError(null);
  };

  if (!supported) {
    return (
      <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit</div>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          Conectar un micro:bit necesita Web Serial, disponible en Chrome o Edge. En este navegador
          podés entrenar igual, pero sin micro:bit.
        </div>
      </div>
    );
  }

  const statusLabel =
    status === "open"
      ? "conectado"
      : status === "connecting"
      ? "conectando"
      : status === "error"
      ? "error"
      : "desconectado";

  return (
    <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit (Web Serial)</div>
      <div style={{ display: "flex", gap: 8 }}>
        {status === "open" ? (
          <button onClick={() => void handleDisconnect()} style={{ flex: 1 }}>
            Desconectar micro:bit
          </button>
        ) : (
          <button
            onClick={() => void handleConnect()}
            disabled={status === "connecting"}
            style={{ flex: 1 }}
          >
            {status === "connecting" ? "Conectando..." : "Conectar micro:bit"}
          </button>
        )}
      </div>
      <div style={{ fontSize: 12 }}>
        Estado: <b>{statusLabel}</b>
      </div>
      <label style={{ fontSize: 12, display: "grid", gap: 4 }}>
        <span>
          Umbral de confianza: <b>{threshold.toFixed(2)}</b>
        </span>
        <input
          type="range"
          min={0.3}
          max={0.95}
          step={0.05}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
      </label>
      {status === "open" && (
        <div style={{ fontSize: 11, fontFamily: "monospace", display: "grid", gap: 2 }}>
          {log.length ? (
            log.map((line, idx) => (
              <div key={idx} style={{ opacity: idx === 0 ? 1 : 0.55 }}>
                {line}
              </div>
            ))
          ) : (
            <div style={{ opacity: 0.55 }}>Sin mensajes todavía.</div>
          )}
        </div>
      )}
      {error && <div style={{ fontSize: 12, color: "#b91c1c" }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.65 }}>
        El micro:bit necesita un programa MakeCode con la extensión SmartTEAM ML. Si MakeCode está
        conectado en otra pestaña, desconectalo antes.
      </div>
    </div>
  );
}
