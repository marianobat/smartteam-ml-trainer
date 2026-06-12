// src/app/components/MicrobitPanel.tsx
//
// Panel "Conectar micro:bit" compartido por todos los entrenadores. Recibe la
// etiqueta estable y su confianza; aplica el umbral y responde cada vez que el
// micro:bit pide la clase actual ("ML?"), por USB (Web Serial) o por
// Bluetooth (Web Bluetooth + UART). Nunca envía nada sin pedido, así el
// buffer del micro:bit no puede llenarse.

import { useEffect, useRef, useState } from "react";
import {
  connectMicrobit,
  disconnectMicrobit,
  isWebSerialSupported,
  sendMicrobitLabel,
} from "../../core/microbit/serialConnection";
import {
  connectMicrobitBle,
  disconnectMicrobitBle,
  isWebBluetoothSupported,
  sendMicrobitLabelBle,
} from "../../core/microbit/bluetoothConnection";
import {
  clearMicrobitListeners,
  setMicrobitListeners,
  type MicrobitTransportKind,
} from "../../core/microbit/transport";
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
  const serialSupported = isWebSerialSupported();
  const bleSupported = isWebBluetoothSupported();
  const [status, setStatus] = useState<PanelStatus>("idle");
  const [transport, setTransport] = useState<MicrobitTransportKind | null>(null);
  const [deviceName, setDeviceName] = useState<string>("");
  const [alias, setAlias] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [log, setLog] = useState<string[]>([]);
  const [requestCount, setRequestCount] = useState(0);

  const statusRef = useRef<PanelStatus>(status);
  statusRef.current = status;
  const transportRef = useRef<MicrobitTransportKind | null>(transport);
  transportRef.current = transport;

  const labelToSendRef = useRef(NONE_LABEL);
  labelToSendRef.current =
    label && label !== NONE_LABEL && confidence >= threshold ? label : NONE_LABEL;

  const pushLog = (line: string) => {
    setLog((prev) => [line, ...prev].slice(0, MAX_LOG_LINES));
  };

  const respondRequest = async () => {
    if (statusRef.current !== "open") return;
    try {
      const line =
        transportRef.current === "bluetooth"
          ? await sendMicrobitLabelBle(labelToSendRef.current)
          : await sendMicrobitLabel(labelToSendRef.current);
      setRequestCount((prev) => prev + 1);
      pushLog(line);
    } catch (err) {
      setStatus("error");
      setTransport(null);
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // Responder cada "ML?" del micro:bit + alias + caída de conexión
  useEffect(() => {
    if (status !== "open") return;
    setMicrobitListeners({
      onRequest: () => {
        void respondRequest();
      },
      onAlias: (incoming) => setAlias(incoming),
      onDrop: () => {
        setStatus("error");
        setTransport(null);
        setError("Se perdió la conexión con la placa. Acercala y volvé a conectar.");
      },
    });
    return () => clearMicrobitListeners();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Desconexión limpia al desmontar
  useEffect(() => {
    return () => {
      clearMicrobitListeners();
      void disconnectMicrobit();
      void disconnectMicrobitBle();
    };
  }, []);

  const resetSession = () => {
    setError(null);
    setRequestCount(0);
    setLog([]);
    setAlias("");
    setDeviceName("");
  };

  const handleConnectSerial = async () => {
    resetSession();
    setStatus("connecting");
    try {
      await connectMicrobit();
      setTransport("serial");
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

  const handleConnectBle = async () => {
    resetSession();
    setStatus("connecting");
    try {
      const name = await connectMicrobitBle();
      setDeviceName(name);
      setTransport("bluetooth");
      setStatus("open");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // El usuario cerró el selector de dispositivos: no es un error
      if (/user cancelled|cancelled the requestdevice|notfounderror/i.test(message)) {
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
      if (transportRef.current === "bluetooth") {
        await disconnectMicrobitBle();
      } else {
        await disconnectMicrobit();
      }
    } finally {
      setTransport(null);
      setStatus("idle");
      setError(null);
    }
  };

  if (!serialSupported && !bleSupported) {
    return (
      <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit</div>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          Conectar un micro:bit necesita Web Serial o Web Bluetooth, disponibles en Chrome o Edge.
          En este navegador podés entrenar igual, pero sin micro:bit.
        </div>
      </div>
    );
  }

  const transportLabel = transport === "bluetooth" ? "Bluetooth" : transport === "serial" ? "USB" : "";
  const statusLabel =
    status === "open"
      ? `conectado por ${transportLabel}`
      : status === "connecting"
      ? "conectando"
      : status === "disconnecting"
      ? "desconectando"
      : status === "error"
      ? "error"
      : "desconectado";

  const connectedName =
    transport === "bluetooth" && (deviceName || alias)
      ? [deviceName, alias].filter(Boolean).join(" · ")
      : "";

  return (
    <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit</div>
      {status === "open" || status === "disconnecting" ? (
        <button
          onClick={() => void handleDisconnect()}
          disabled={status === "disconnecting"}
          style={{ flex: 1 }}
        >
          {status === "disconnecting" ? "Desconectando..." : "Desconectar micro:bit"}
        </button>
      ) : (
        <div style={{ display: "flex", gap: 8 }}>
          {serialSupported && (
            <button
              onClick={() => void handleConnectSerial()}
              disabled={status === "connecting"}
              style={{ flex: 1 }}
            >
              {status === "connecting" ? "Conectando..." : "🔌 USB"}
            </button>
          )}
          {bleSupported && (
            <button
              onClick={() => void handleConnectBle()}
              disabled={status === "connecting"}
              style={{ flex: 1 }}
            >
              {status === "connecting" ? "Conectando..." : "📶 Bluetooth"}
            </button>
          )}
        </div>
      )}
      <div style={{ fontSize: 12 }}>
        Estado: <b>{statusLabel}</b>
        {status === "open" && connectedName && (
          <>
            {" "}
            — placa: <b>{connectedName}</b>
          </>
        )}
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
        USB: extensión SmartTEAM ML / Bluetooth: extensión SmartTEAM ML Bluetooth. 
        Si MakeCode está conectado a la placa en otra pestaña, desconectar antes.
      </div>
    </div>
  );
}
