// src/app/hooks/useMicrobit.ts
//
// Store compartido de la conexión micro:bit. Antes esta lógica vivía dentro de
// MicrobitPanel; ahora es un singleton a nivel módulo para que la conexión y el
// "responder a pedido" (ML?) funcionen sin importar desde dónde se haya
// conectado — el panel de la página o la ventana flotante (PiP). Como la
// conexión física ya es única (un solo puerto/placa a la vez), el estado y los
// listeners también son únicos.

import { useSyncExternalStore } from "react";
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
import { setMicrobitListeners, type MicrobitTransportKind } from "../../core/microbit/transport";
import { DEFAULT_CONFIDENCE_THRESHOLD, NONE_LABEL } from "../../core/microbit/protocol";
import { COPY } from "../copy";

export type MicrobitStatus = "idle" | "connecting" | "open" | "disconnecting" | "error";

export type MicrobitState = {
  status: MicrobitStatus;
  transport: MicrobitTransportKind | null;
  deviceName: string;
  alias: string;
  error: string | null;
  threshold: number;
  requestCount: number;
  log: string[];
};

const MAX_LOG_LINES = 6;

let state: MicrobitState = {
  status: "idle",
  transport: null,
  deviceName: "",
  alias: "",
  error: null,
  threshold: DEFAULT_CONFIDENCE_THRESHOLD,
  requestCount: 0,
  log: [],
};

// Última detección estable (no provoca re-render: se lee solo al responder).
let currentLabel = NONE_LABEL;
let currentConfidence = 0;

const subscribers = new Set<() => void>();
let listenersWired = false;

function emit() {
  subscribers.forEach((fn) => fn());
}

function setState(patch: Partial<MicrobitState>) {
  state = { ...state, ...patch };
  emit();
}

function labelToSend(): string {
  return currentLabel && currentLabel !== NONE_LABEL && currentConfidence >= state.threshold
    ? currentLabel
    : NONE_LABEL;
}

async function respondRequest() {
  if (state.status !== "open") return;
  try {
    const line =
      state.transport === "bluetooth"
        ? await sendMicrobitLabelBle(labelToSend())
        : await sendMicrobitLabel(labelToSend());
    setState({
      requestCount: state.requestCount + 1,
      log: [line, ...state.log].slice(0, MAX_LOG_LINES),
    });
  } catch (err) {
    setState({
      status: "error",
      transport: null,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

// Se cablea una sola vez (al primer connect). Los listeners son globales y
// quedan activos toda la vida de la pestaña; respondRequest filtra por estado.
function wireListeners() {
  if (listenersWired) return;
  listenersWired = true;
  setMicrobitListeners({
    onRequest: () => {
      void respondRequest();
    },
    onAlias: (incoming) => setState({ alias: incoming }),
    onDrop: () =>
      setState({
        status: "error",
        transport: null,
        error: COPY.mbLostConnection,
      }),
  });
}

function resetSession() {
  setState({ error: null, requestCount: 0, log: [], alias: "", deviceName: "" });
}

async function connectUsb() {
  resetSession();
  setState({ status: "connecting" });
  wireListeners();
  try {
    await connectMicrobit();
    setState({ transport: "serial", status: "open" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // El usuario canceló el diálogo de puertos: no es un error.
    if (/no port selected/i.test(message)) {
      setState({ status: "idle" });
      return;
    }
    setState({ status: "error", error: message });
  }
}

async function connectBle() {
  resetSession();
  setState({ status: "connecting" });
  wireListeners();
  try {
    const name = await connectMicrobitBle();
    setState({ deviceName: name, transport: "bluetooth", status: "open" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // El usuario cerró el selector de dispositivos: no es un error.
    if (/user cancelled|cancelled the requestdevice|notfounderror/i.test(message)) {
      setState({ status: "idle" });
      return;
    }
    setState({ status: "error", error: message });
  }
}

async function disconnect() {
  setState({ status: "disconnecting" });
  try {
    if (state.transport === "bluetooth") {
      await disconnectMicrobitBle();
    } else {
      await disconnectMicrobit();
    }
  } finally {
    setState({ transport: null, status: "idle", error: null });
  }
}

function setThreshold(value: number) {
  setState({ threshold: value });
}

/** El entrenador empuja la detección estable actual (clase + confianza). */
function setCurrentDetection(label: string, confidence: number) {
  currentLabel = label || NONE_LABEL;
  currentConfidence = confidence;
}

/** Snapshot vivo (para el polling de la ventana flotante, fuera de React). */
function getMicrobitState(): MicrobitState {
  return state;
}

export const microbitApi = {
  supported: { serial: isWebSerialSupported(), bluetooth: isWebBluetoothSupported() },
  connectUsb,
  connectBle,
  disconnect,
  setThreshold,
  setCurrentDetection,
  getState: getMicrobitState,
};

export type MicrobitApi = typeof microbitApi;

function subscribe(callback: () => void) {
  subscribers.add(callback);
  return () => {
    subscribers.delete(callback);
  };
}

/** Hook React: estado reactivo + acciones del micro:bit (estado compartido). */
export function useMicrobit() {
  const snapshot = useSyncExternalStore(subscribe, getMicrobitState, getMicrobitState);
  return {
    ...snapshot,
    supported: microbitApi.supported,
    connectUsb,
    connectBle,
    disconnect,
    setThreshold,
    setCurrentDetection,
  };
}
