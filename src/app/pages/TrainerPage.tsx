import { useState } from "react";
import Trainer, { type TrainerConfig } from "./Trainer";
import TextTrainer from "./TextTrainer";
import AudioTrainer from "./AudioTrainer";
import { createHandExtractor } from "../../core/extractors/handExtractor";
import { createPoseExtractor } from "../../core/extractors/poseExtractor";
import { createFaceExtractor } from "../../core/extractors/faceExtractor";
import { createImageExtractor } from "../../core/extractors/imageExtractor";
import { TURBOWARP_ENABLED } from "../../core/bridge/features";
import { getToken, getRoom } from "../../core/bridge/session";
import "./TrainerPage.css";

const coverUrl = (name: string) => `${import.meta.env.BASE_URL}covers/${name}.svg`;

const getRoomFromQuery = () => {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  return params.get("room") ?? "";
};

type ModelId = "hands" | "face" | "images" | "pose" | "text" | "audio";

const MODEL_IDS: readonly ModelId[] = ["hands", "face", "images", "pose", "text", "audio"];

/** Permite entrar directo a una modalidad con ?model= (p. ej. desde micro:bit). */
const getModelFromQuery = (): ModelId | null => {
  if (typeof window === "undefined") return null;
  const value = new URLSearchParams(window.location.search).get("model");
  return MODEL_IDS.find((id) => id === value) ?? null;
};

const trainerConfigs: Partial<Record<ModelId, TrainerConfig>> = {
  hands: {
    title: "Entrenador de manos",
    loadingText: "Preparando el detector de manos...",
    missingLabel: "Sin manos",
    missingHint: "No veo tus manos. Ponlas frente a la cámara",
    placeholderIcon: "✋",
    thumbnailSource: "overlay",
    storageKey: "hands",
    createExtractor: createHandExtractor,
  },
  pose: {
    title: "Entrenador de posturas",
    loadingText: "Preparando el detector de cuerpo...",
    missingLabel: "Sin cuerpo",
    missingHint: "No te veo. Aléjate un poco de la cámara",
    placeholderIcon: "🧍",
    thumbnailSource: "overlay",
    storageKey: "pose",
    createExtractor: createPoseExtractor,
  },
  face: {
    title: "Entrenador de caras",
    loadingText: "Preparando el detector de caras...",
    missingLabel: "Sin cara",
    missingHint: "No veo tu cara. Acércate a la cámara",
    placeholderIcon: "😀",
    thumbnailSource: "overlay",
    storageKey: "face",
    createExtractor: createFaceExtractor,
  },
  images: {
    title: "Entrenador de imágenes",
    loadingText: "Preparando el detector de imágenes...",
    missingLabel: "Sin imagen",
    missingHint: "Muéstrale algo a la cámara",
    placeholderIcon: "🖼️",
    thumbnailSource: "video",
    storageKey: "images",
    createExtractor: createImageExtractor,
  },
};

export default function TrainerPage() {
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const room = getRoomFromQuery() || getRoom() || "";
  const publishToken = getToken() || "";
  const [selectedModel, setSelectedModel] = useState<ModelId | null>(getModelFromQuery);

  const models = [
    {
      id: "hands",
      title: "Manos",
      description: "Gestos y señas con tus manos.",
      enabled: true,
      cover: coverUrl("hands"),
    },
    {
      id: "face",
      title: "Caras",
      description: "Sonrisas, guiños y expresiones.",
      enabled: true,
      cover: coverUrl("face"),
    },
    {
      id: "pose",
      title: "Cuerpo",
      description: "Posturas y movimientos enteros.",
      enabled: true,
      cover: coverUrl("pose"),
    },
    {
      id: "images",
      title: "Imágenes",
      description: "Objetos y dibujos frente a la cámara.",
      enabled: true,
      cover: coverUrl("images"),
    },
    {
      id: "text",
      title: "Textos",
      description: "Frases y palabras escritas.",
      enabled: true,
      cover: coverUrl("text"),
    },
    {
      id: "audio",
      title: "Sonidos",
      description: "Palabras y sonidos con el micrófono.",
      enabled: true,
      cover: coverUrl("audio"),
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
            <h1 className="trainer-select-title">¿Qué le quieres enseñar?</h1>
            <p className="trainer-select-subtitle">
              Elige una modalidad para entrenar tu modelo con ejemplos.
            </p>
          </div>
          {TURBOWARP_ENABLED && room && (
            <div className="trainer-select-room">Room: {room}</div>
          )}
        </header>
        {TURBOWARP_ENABLED && !room && (
          <div className="trainer-select-warning">
            Sin sesión de TurboWarp: podés entrenar y usar micro:bit igual. Para publicar a
            TurboWarp, crea una sesión en el lobby.
          </div>
        )}
        <section className="trainer-select-grid">
          {models.map((model) => {
            const disabled = !model.enabled;
            return (
              <button
                key={model.id}
                type="button"
                className={`model-card model-card--${model.id} ${disabled ? "is-disabled" : ""}`}
                onClick={() => {
                  if (!disabled) {
                    setSelectedModel(model.id);
                  }
                }}
                disabled={disabled}
              >
                <div className="model-card-media" aria-hidden="true">
                  <img
                    className="model-card-cover"
                    src={model.cover}
                    alt=""
                    width={400}
                    height={240}
                    loading="lazy"
                    decoding="async"
                  />
                </div>
                <div className="model-card-body">
                  <div className="model-card-title">{model.title}</div>
                  <div className="model-card-meta">{model.description}</div>
                </div>
              </button>
            );
          })}
        </section>
        {TURBOWARP_ENABLED && (
          <div className="trainer-select-actions">
            <button type="button" onClick={() => window.location.assign(baseUrl)}>
              Volver al Lobby
            </button>
          </div>
        )}
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
