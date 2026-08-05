// src/app/components/MicrobitPanel.tsx
//
// Panel "Conectar micro:bit" compartido por todos los entrenadores. Es una
// vista delgada sobre el store useMicrobit: muestra estado y botones de
// conexión (USB / Bluetooth), aplica el umbral en modo avanzado y deja el log.
// La lógica de conexión y de "responder a pedido" (ML?) vive en el store, así
// que conectar desde la ventana flotante (PiP) funciona igual.

import { useEffect } from "react";
import { Usb, Bluetooth } from "lucide-react";
import { useMicrobit } from "../hooks/useMicrobit";
import { COPY } from "../copy";

/** USB queda en el código pero oculto: por ahora solo ofrecemos Bluetooth. */
const SHOW_USB_CONNECT = false;

type MicrobitPanelProps = {
  /** Etiqueta estable actual ("" o "none" si no hay detección). */
  label: string;
  confidence: number;
  /** Modo avanzado: muestra umbral y log de mensajes. */
  advanced?: boolean;
};

export default function MicrobitPanel({ label, confidence, advanced = false }: MicrobitPanelProps) {
  const mb = useMicrobit();
  const { supported } = mb;

  // Empujamos la detección estable al store: alimenta el responder de "ML?".
  // setCurrentDetection es una función estable del módulo (no recrea el efecto).
  useEffect(() => {
    mb.setCurrentDetection(label, confidence);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, confidence]);

  // Desconexión limpia al desmontar (salir del entrenador).
  useEffect(() => {
    return () => {
      void mb.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // El panel de conexión (Bluetooth/estado) vive solo en modo avanzado. Los
  // efectos de arriba (empujar la detección al store, desconectar al desmontar)
  // corren igual, así la conexión persiste aunque se cierre el modo avanzado.
  if (!advanced) return null;

  if (!supported.serial && !supported.bluetooth) {
    return (
      <div style={{ paddingTop: 10, display: "grid", gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit</div>
        <div style={{ fontSize: 12, opacity: 0.75 }}>{COPY.mbNoBluetooth}</div>
      </div>
    );
  }

  const { status, transport, deviceName, alias, error, requestCount, threshold, log } = mb;
  const transportLabel =
    transport === "bluetooth" ? "Bluetooth" : transport === "serial" ? "USB" : "";
  const statusLabel =
    status === "open"
      ? COPY.mbConnectedVia(transportLabel)
      : status === "connecting"
      ? COPY.mbStateConnecting
      : status === "disconnecting"
      ? COPY.mbStateDisconnecting
      : status === "error"
      ? COPY.mbStateError
      : COPY.mbStateDisconnected;

  const connectedName =
    transport === "bluetooth" && (deviceName || alias)
      ? [deviceName, alias].filter(Boolean).join(" · ")
      : "";

  const buttonStyle = {
    flex: 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  } as const;

  return (
    <div style={{ paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>micro:bit</div>
      {status === "open" || status === "disconnecting" ? (
        <button
          onClick={() => void mb.disconnect()}
          disabled={status === "disconnecting"}
          style={{ width: "100%" }}
        >
          {status === "disconnecting" ? COPY.mbDisconnecting : COPY.mbDisconnect}
        </button>
      ) : (
        <div style={{ display: "flex", gap: 8 }}>
          {SHOW_USB_CONNECT && supported.serial && (
            <button
              onClick={() => void mb.connectUsb()}
              disabled={status === "connecting"}
              style={buttonStyle}
            >
              {status === "connecting" ? (
                COPY.mbConnecting
              ) : (
                <>
                  <Usb size={16} aria-hidden="true" /> USB
                </>
              )}
            </button>
          )}
          {supported.bluetooth && (
            <button
              onClick={() => void mb.connectBle()}
              disabled={status === "connecting"}
              style={{ ...buttonStyle, width: "100%" }}
            >
              {status === "connecting" ? (
                COPY.mbConnecting
              ) : (
                <>
                  <Bluetooth size={16} aria-hidden="true" /> Bluetooth
                </>
              )}
            </button>
          )}
        </div>
      )}
      <div style={{ fontSize: 12 }}>
        {COPY.advStatus} <b>{statusLabel}</b>
        {status === "open" && connectedName && (
          <>
            {" "}
            — {COPY.mbBoard} <b>{connectedName}</b>
          </>
        )}
        {status === "open" && (
          <>
            {" "}
            — {COPY.mbRequests} <b>{requestCount}</b>
          </>
        )}
      </div>
      {advanced && (
        <label style={{ fontSize: 12, display: "grid", gap: 4 }}>
          <span>
            {COPY.mbThreshold} <b>{threshold.toFixed(2)}</b>
          </span>
          <input
            type="range"
            min={0.3}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(e) => mb.setThreshold(Number(e.target.value))}
            style={{ accentColor: "var(--color-secondary)" }}
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
            <div style={{ opacity: 0.55 }}>{COPY.mbWaiting}</div>
          )}
        </div>
      )}
      {error && <div style={{ fontSize: 12, color: "#b91c1c" }}>{error}</div>}
    </div>
  );
}
