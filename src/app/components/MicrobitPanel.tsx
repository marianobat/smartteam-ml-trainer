// src/app/components/MicrobitPanel.tsx
//
// Panel "Conectar micro:bit" compartido por todos los entrenadores. Recibe la
// etiqueta estable y su confianza; aplica el umbral y publica por Web Serial.
//
// Dos modos:
//  - "a pedido" (default): responde una línea solo cuando el micro:bit envía
//    "ML?" (bloque "pedir clase ML"). No puede llenar el buffer del micro:bit.
//  - "automático": empuja al cambiar la etiqueta + heartbeat (compatibilidad
//    con programas viejos que solo usan "al detectar clase ML").

import { useEffect, useRef, useState } from "react";
import {
  connectMicrobit,
  disconnectMicrobit,
  isWebSerialSupported,
  sendMicrobitLabel,
  setMicrobitRequestListener,
} from "../../core/microbit/serialConnection";
import {
  DEFAULT_CONFIDENCE_THRESHOLD,
  NONE_LABEL,
  RESEND_INTERVAL_MS,
} from "../../core/microbit/protocol";

const MAX_LOG_LINES = 6;

type PanelStatus = "idle" | "connecting" | "open" | "error";
type SendMode = "ondemand" | "auto";

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
  const [sendMode, setSendMode] = useState<SendMode>("ondemand");
  const [log, setLog] = useState<string[]>([]);
  const [requestCount, setRequestCount] = useState(0);

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

  // Modo "a pedido": responder cada "ML?" del micro:bit
  useEffect(() => {
    if (status !== "open" || sendMode !== "ondemand") return;
    setMicrobitRequestListener(() => {
      setRequestCount((prev) => prev + 1);
      void trySend(true);
    });
    return () => setMicrobitRequestListener(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, sendMode]);

  // Modo "automático": envío inmediato cuando cambia la etiqueta efectiva
  useEffect(() => {
    if (sendMode !== "auto") return;
    void trySend(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, confidence, threshold, status, sendMode]);

  // Modo "automático": heartbeat cada RESEND_INTERVAL_MS
  useEffect(() => {
    if (status !== "open" || sendMode !== "auto") return;
    const id = window.setInterval(() => {
      void trySend(true);
    }, RESEND_INTERVAL_MS);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, sendMode]);

  // Desconexión limpia al desmontar
  useEffect(() => {
    return () => {
      setMicrobitRequestListener(null);
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
        {status === "open" && sendMode === "ondemand" && (
          <>
            {" "}
            — pedidos respondidos: <b>{requestCount}</b>
          </>
        )}
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          type="button"
          onClick={() => setSendMode("ondemand")}
          style={{
            flex: 1,
            fontSize: 12,
            borderRadius: 8,
            border: sendMode === "ondemand" ? "2px solid #111" : "1px solid #ddd",
            background: sendMode === "ondemand" ? "#111" : "#fff",
            color: sendMode === "ondemand" ? "#fff" : "#111",
          }}
        >
          A pedido
        </button>
        <button
          type="button"
          onClick={() => setSendMode("auto")}
          style={{
            flex: 1,
            fontSize: 12,
            borderRadius: 8,
            border: sendMode === "auto" ? "2px solid #111" : "1px solid #ddd",
            background: sendMode === "auto" ? "#111" : "#fff",
            color: sendMode === "auto" ? "#fff" : "#111",
          }}
        >
          Automático
        </button>
      </div>
      <div style={{ fontSize: 11, opacity: 0.65 }}>
        {sendMode === "ondemand"
          ? 'Responde solo cuando el programa usa el bloque "pedir clase ML". Evita que se llene el buffer del micro:bit.'
          : "Envía la clase al cambiar y cada 500 ms. Solo para programas viejos; puede colgar el micro:bit si su programa se bloquea."}
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
