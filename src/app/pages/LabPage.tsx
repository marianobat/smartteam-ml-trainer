// src/app/pages/LabPage.tsx
//
// Página de laboratorio (/lab) — PRUEBA, aislada del resto. Muestra la
// evaluación en vivo de un modelo YA entrenado (cámara + efectos + barras por
// clase, como la ventana PiP) junto a un MakeCode micro:bit embebido y
// editable. La conexión Bluetooth/USB usa el store compartido useMicrobit, así
// el micro:bit responde a "ML?" con lo que el modelo detecta.
//
// No entrena ni captura: carga el modelo guardado en IndexedDB por modalidad
// (?model=hands|face|pose|images, default hands).

import { useRef, useState } from "react";
import { Bluetooth, Usb, ArrowLeft } from "lucide-react";
import { createHandExtractor } from "../../core/extractors/handExtractor";
import { createPoseExtractor } from "../../core/extractors/poseExtractor";
import { createFaceExtractor } from "../../core/extractors/faceExtractor";
import { createImageExtractor } from "../../core/extractors/imageExtractor";
import CameraStage from "../components/trainer/CameraStage";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import MakeCodeEmbed from "../components/MakeCodeEmbed";
import { useLiveEvaluation, type EvalConfig } from "../hooks/useLiveEvaluation";
import { useMicrobit } from "../hooks/useMicrobit";
import "./LabPage.css";

/** USB queda en el código pero oculto: por ahora solo ofrecemos Bluetooth. */
const SHOW_USB_CONNECT = false;

type ModelId = "hands" | "face" | "pose" | "images";

const CONFIGS: Record<ModelId, EvalConfig & { label: string }> = {
  hands: {
    label: "Manos",
    storageKey: "hands",
    missingLabel: "Sin manos",
    dimmed: true,
    createExtractor: createHandExtractor,
  },
  face: {
    label: "Caras",
    storageKey: "face",
    missingLabel: "Sin cara",
    dimmed: true,
    createExtractor: createFaceExtractor,
  },
  pose: {
    label: "Cuerpo",
    storageKey: "pose",
    missingLabel: "Sin cuerpo",
    dimmed: true,
    createExtractor: createPoseExtractor,
  },
  images: {
    label: "Imágenes",
    storageKey: "images",
    missingLabel: "Sin imagen",
    dimmed: false,
    createExtractor: createImageExtractor,
  },
};

const getInitialModel = (): ModelId => {
  if (typeof window === "undefined") return "hands";
  const param = new URLSearchParams(window.location.search).get("model");
  return param && param in CONFIGS ? (param as ModelId) : "hands";
};

export default function LabPage() {
  const [model, setModel] = useState<ModelId>(getInitialModel);
  const baseUrl = import.meta.env.BASE_URL ?? "/";

  return (
    <div className="lab-page">
      <header className="lab-header">
        <a className="lab-back" href={`${baseUrl}trainer`}>
          <ArrowLeft size={16} aria-hidden="true" /> Entrenador
        </a>
        <h1 className="lab-title">Laboratorio</h1>
        <div className="lab-model-switch" role="group" aria-label="Modalidad">
          {(Object.keys(CONFIGS) as ModelId[]).map((id) => (
            <button
              key={id}
              type="button"
              className={`lab-model-btn ${id === model ? "is-on" : ""}`}
              aria-pressed={id === model}
              onClick={() => setModel(id)}
            >
              {CONFIGS[id].label}
            </button>
          ))}
        </div>
      </header>

      <div className="lab-main">
        <section className="lab-eval">
          {/* key=model: remonta cámara + extractor + modelo al cambiar modalidad */}
          <LiveEvalColumn key={model} config={CONFIGS[model]} baseUrl={baseUrl} />
        </section>
        <section className="lab-embed">
          <MakeCodeEmbed />
        </section>
      </div>
    </div>
  );
}

function LiveEvalColumn({ config, baseUrl }: { config: EvalConfig; baseUrl: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const evaluation = useLiveEvaluation(videoRef, canvasRef, config);
  const mb = useMicrobit();

  const cameraHint =
    !evaluation.loading && !evaluation.hasSubject ? config.missingLabel : null;

  const connected = mb.status === "open";
  const connecting = mb.status === "connecting";

  return (
    <>
      <CameraStage
        videoRef={videoRef}
        canvasRef={canvasRef}
        dimmed={config.dimmed}
        loading={evaluation.loading}
        loadingText={evaluation.status}
        hint={cameraHint}
      />

      {!evaluation.hasModel && !evaluation.loading && (
        <div className="lab-no-model">
          No hay un modelo entrenado para esta modalidad en este navegador.{" "}
          <a href={`${baseUrl}trainer`}>Entrená uno primero</a>.
        </div>
      )}

      <LivePredictionBars
        rows={evaluation.rows}
        seeing={evaluation.seeing}
        hasModel={evaluation.hasModel}
      />

      <div className="lab-microbit">
        {connected ? (
          <button type="button" className="lab-mb-disconnect" onClick={() => void mb.disconnect()}>
            Desconectar micro:bit
          </button>
        ) : (
          <div className="lab-mb-buttons">
            {mb.supported.bluetooth && (
              <button
                type="button"
                className="lab-mb-connect"
                disabled={connecting}
                onClick={() => void mb.connectBle()}
              >
                <Bluetooth size={16} aria-hidden="true" /> {connecting ? "Conectando..." : "Bluetooth"}
              </button>
            )}
            {SHOW_USB_CONNECT && mb.supported.serial && (
              <button
                type="button"
                className="lab-mb-connect"
                disabled={connecting}
                onClick={() => void mb.connectUsb()}
              >
                <Usb size={16} aria-hidden="true" /> {connecting ? "Conectando..." : "USB"}
              </button>
            )}
          </div>
        )}
        <div className="lab-mb-status">
          {connected
            ? `micro:bit conectado (${mb.transport === "bluetooth" ? "Bluetooth" : "USB"})`
            : mb.status === "error"
            ? mb.error ?? "Error de conexión"
            : "micro:bit desconectado"}
        </div>
      </div>
    </>
  );
}
