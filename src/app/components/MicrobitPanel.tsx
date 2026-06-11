// src/app/components/MicrobitPanel.tsx
//
// Panel "Conectar micro:bit" compartido por todos los entrenadores. Recibe la
// etiqueta estable y su confianza; aplica el umbral y responde por Web Serial
// cada vez que el micro:bit pide la clase actual ("ML?"). Nunca envía nada
// sin pedido, así el buffer del micro:bit no puede llenarse.

import { useEffect, useRef, useState } from "react";
import {
  connectMicrobit,
  disconnectMicrobit,
  isWebSerialSupported,
  sendMicrobitLabel,
  setMicrobitRequestListener,
} from "../../core/microbit/serialConnection";
import { DEFAULT_CONFIDENCE_THRESHOLD, NONE_LABEL } from "../../core/microbit/protocol";

const MAX_LOG_LINES = 6;

type PanelStatus = "idle" | "connecting" | "open" | "disconnecting" | "error";

type MicrobitPanelProps = {
  /** Etiqueta estable actual ("" o NONE_LABEL si no hay detección). */
  label: string;
  confidence: number;
  /** Modo avanzado: muestra umbral y log de mensajes. */
  advanced?: boolean;
};

export default function MicrobitPanel({ label, confidence, advanced = false }: MicrobitPanelProps) {
  const supported = isWebSerialSupported();
  const [status, setStatus] = useState<PanelStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [log, setLog] = useState<string[]>([]);
  const [requestCount, setRequestCount] = useState(0);

  const statusRef = useRef<PanelStatus>(status);
  statusRef.current = status;

  const labelToSendRef = useRef(NONE_LABEL);
  labelToSendRef.current =
    label && label !== NONE_LABEL && confidence >= threshold ? label : NONE_LABEL;

  const pushLog = (line: string) => {
    setLog((prev) => [line, ...prev].slice(0, MAX_LOG_LINES));
  };

  const respondRequest = async () => {
    if (statusRef.current !== "open") return;
    try {
      const line = await sendMicrobitLabel(labelToSendRef.current);
      setRequestCount((prev) => prev + 1);
      pushLog(line);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // Responder cada "ML?" del micro:bit
  useEffect(() => {
    if (status !== "open") return;
    setMicrobitRequestListener(() => {
      void respondRequest();
    });
    return () => setMicrobitRequestListener(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

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
    setRequestCount(0);
    setLog([]);
    try {
      await connectMicrobit();
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
    setStatus("disconnecting");
    try {
      await disconnectMicrobit();
    } finally {
      setStatus("idle");
      setError(null);
    }
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
      : status === "disconnecting"
      ? "desconectando"
      : status === "error"
      ? "error"
      : "desconectado";

  return (
    <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit (Web Serial)</div>
      <div style={{ display: "flex", gap: 8 }}>
        {status === "open" || status === "disconnecting" ? (
          <button
            onClick={() => void handleDisconnect()}
            disabled={status === "disconnecting"}
            style={{ flex: 1 }}
          >
            {status === "disconnecting" ? "Desconectando..." : "Desconectar micro:bit"}
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
        {status === "open" && (
          <>
            {" "}
            — pedidos respondidos: <b>{requestCount}</b>
          </>
        )}
      </div>
      {advanced && (
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
      )}
      {advanced && status === "open" && (
        <div style={{ fontSize: 11, fontFamily: "monospace", display: "grid", gap: 2 }}>
          {log.length ? (
            log.map((line, idx) => (
              <div key={idx} style={{ opacity: idx === 0 ? 1 : 0.55 }}>
                {line}
              </div>
            ))
          ) : (
            <div style={{ opacity: 0.55 }}>Esperando pedidos del micro:bit...</div>
          )}
        </div>
      )}
      {error && <div style={{ fontSize: 12, color: "#b91c1c" }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.65 }}>
        Responde solo cuando el micro:bit pregunta (extensión SmartTEAM ML v0.3+). Si MakeCode está
        conectado en otra pestaña, desconectalo antes.
      </div>
    </div>
  );
}
