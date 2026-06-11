import { useState } from "react";
import Trainer, { type TrainerConfig } from "./Trainer";
import TextTrainer from "./TextTrainer";
import AudioTrainer from "./AudioTrainer";
import { createHandExtractor } from "../../core/extractors/handExtractor";
import { createPoseExtractor } from "../../core/extractors/poseExtractor";
import { createFaceExtractor } from "../../core/extractors/faceExtractor";
import { createImageExtractor } from "../../core/extractors/imageExtractor";
import { getToken, getRoom } from "../../core/bridge/session";
import "./TrainerPage.css";

const getRoomFromQuery = () => {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  return params.get("room") ?? "";
};

type ModelId = "hands" | "face" | "images" | "pose" | "text" | "audio";

const trainerConfigs: Partial<Record<ModelId, TrainerConfig>> = {
  hands: {
    title: "Entrenador de manos (2 manos)",
    loadingText: "Cargando modelo de manos...",
    missingLabel: "Sin manos",
    storageKey: "hands",
    createExtractor: createHandExtractor,
  },
  pose: {
    title: "Entrenador de postura corporal",
    loadingText: "Cargando modelo de cuerpo...",
    missingLabel: "Sin cuerpo",
    storageKey: "pose",
    createExtractor: createPoseExtractor,
  },
  face: {
    title: "Entrenador de gestos de la cara",
    loadingText: "Cargando modelo de rostro...",
    missingLabel: "Sin cara",
    storageKey: "face",
    createExtractor: createFaceExtractor,
  },
  images: {
    title: "Entrenador de imagenes",
    loadingText: "Cargando MobileNet...",
    missingLabel: "Sin imagen",
    storageKey: "images",
    createExtractor: createImageExtractor,
  },
};

export default function TrainerPage() {
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const room = getRoomFromQuery() || getRoom() || "";
  const publishToken = getToken() || "";
  const [selectedModel, setSelectedModel] = useState<ModelId | null>(null);

  const models = [
    {
      id: "hands",
      title: "Gesto de las manos",
      description: "Camara + MediaPipe + clasificador en vivo.",
      enabled: true,
      imageLabel: "Manos",
    },
    {
      id: "face",
      title: "Gesto de la cara",
      description: "Expresiones y movimiento facial.",
      enabled: true,
      imageLabel: "Cara",
    },
    {
      id: "images",
      title: "Imagenes",
      description: "Reconocer objetos o escenas.",
      enabled: true,
      imageLabel: "Imagen",
    },
    {
      id: "pose",
      title: "Postura del cuerpo",
      description: "Pose completa con articulaciones.",
      enabled: true,
      imageLabel: "Cuerpo",
    },
    {
      id: "text",
      title: "Textos",
      description: "Clasificacion y comandos por texto.",
      enabled: true,
      imageLabel: "Texto",
    },
    {
      id: "audio",
      title: "Sonidos",
      description: "Palabras y sonidos por microfono.",
      enabled: true,
      imageLabel: "Audio",
    },
  ] as const;

  if (selectedModel === "text") {
    return <TextTrainer onBack={() => setSelectedModel(null)} room={room} publishToken={publishToken} />;
  }
  if (selectedModel === "audio") {
    return <AudioTrainer onBack={() => setSelectedModel(null)} room={room} publishToken={publishToken} />;
  }

  const activeConfig = selectedModel ? trainerConfigs[selectedModel] : undefined;

  if (!activeConfig) {
    return (
      <div className="trainer-select">
        <header className="trainer-select-header">
          <div>
            <div className="trainer-select-kicker">SmartTEAM IA</div>
            <h1 className="trainer-select-title">Selecciona un modelo</h1>
            <p className="trainer-select-subtitle">
              Elegi que queres entrenar: manos, cara, cuerpo, imagenes, textos o sonidos.
            </p>
          </div>
          <div className="trainer-select-room">Room: {room || "—"}</div>
        </header>
        {!room && (
          <div className="trainer-select-warning">
            Sin sesion de TurboWarp: podes entrenar y usar micro:bit igual. Para publicar a
            TurboWarp, crea una sesion en el lobby.
          </div>
        )}
        <section className="trainer-select-grid">
          {models.map((model) => {
            const disabled = !model.enabled;
            return (
              <button
                key={model.id}
                type="button"
                className={`model-card ${disabled ? "is-disabled" : ""}`}
                onClick={() => {
                  if (!disabled) {
                    setSelectedModel(model.id);
                  }
                }}
                disabled={disabled}
              >
                <div className={`model-card-media model-card-media--${model.id}`}>
                  <span>{model.imageLabel}</span>
                </div>
                <div className="model-card-body">
                  <div className="model-card-title">{model.title}</div>
                  <div className="model-card-meta">{model.description}</div>
                  <div className="model-card-status">{disabled ? "Proximamente" : "Disponible"}</div>
                </div>
              </button>
            );
          })}
        </section>
        <div className="trainer-select-actions">
          <button type="button" onClick={() => window.location.assign(baseUrl)}>
            Volver al Lobby
          </button>
        </div>
      </div>
    );
  }

  return (
    <Trainer
      key={selectedModel}
      config={activeConfig}
      onBack={() => setSelectedModel(null)}
      room={room}
      publishToken={publishToken}
    />
  );
}
