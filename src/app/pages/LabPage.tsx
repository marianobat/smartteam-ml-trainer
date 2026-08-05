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
import { COPY } from "../copy";
import "./LabPage.css";

/** USB queda en el código pero oculto: por ahora solo ofrecemos Bluetooth. */
const SHOW_USB_CONNECT = false;

type ModelId = "hands" | "face" | "pose" | "images";

const CONFIGS: Record<ModelId, EvalConfig & { label: string }> = {
  hands: {
    label: COPY.modalities.hands.label,
    storageKey: "hands",
    missingLabel: COPY.modalities.hands.missingLabel,
    dimmed: true,
    createExtractor: createHandExtractor,
  },
  face: {
    label: COPY.modalities.face.label,
    storageKey: "face",
    missingLabel: COPY.modalities.face.missingLabel,
    dimmed: true,
    createExtractor: createFaceExtractor,
  },
  pose: {
    label: COPY.modalities.pose.label,
    storageKey: "pose",
    missingLabel: COPY.modalities.pose.missingLabel,
    dimmed: true,
    createExtractor: createPoseExtractor,
  },
  images: {
    label: COPY.modalities.images.label,
    storageKey: "images",
    missingLabel: COPY.modalities.images.missingLabel,
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
          <ArrowLeft size={16} aria-hidden="true" /> {COPY.labBack}
        </a>
        <h1 className="lab-title">{COPY.labTitle}</h1>
        <div className="lab-model-switch" role="group" aria-label={COPY.ariaModality}>
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
          {COPY.noModelModality} <a href={`${baseUrl}trainer`}>{COPY.trainFirst}</a>.
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
            {COPY.mbDisconnect}
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
                <Bluetooth size={16} aria-hidden="true" /> {connecting ? COPY.mbConnecting : "Bluetooth"}
              </button>
            )}
            {SHOW_USB_CONNECT && mb.supported.serial && (
              <button
                type="button"
                className="lab-mb-connect"
                disabled={connecting}
                onClick={() => void mb.connectUsb()}
              >
                <Usb size={16} aria-hidden="true" /> {connecting ? COPY.mbConnecting : "USB"}
              </button>
            )}
          </div>
        )}
        <div className="lab-mb-status">
          {connected
            ? COPY.mbConnected(mb.transport === "bluetooth" ? "Bluetooth" : "USB")
            : mb.status === "error"
            ? mb.error ?? COPY.mbConnectionError
            : COPY.mbDisconnected}
        </div>
      </div>
    </>
  );
}
