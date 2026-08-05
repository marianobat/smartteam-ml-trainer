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
import { COPY } from "../copy";
import LangSwitch from "../components/LangSwitch";
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
    title: COPY.modalities.hands.trainerTitle,
    loadingText: COPY.modalities.hands.loadingText,
    missingLabel: COPY.modalities.hands.missingLabel,
    missingHint: COPY.modalities.hands.missingHint,
    placeholderIcon: "✋",
    thumbnailSource: "overlay",
    storageKey: "hands",
    createExtractor: createHandExtractor,
  },
  pose: {
    title: COPY.modalities.pose.trainerTitle,
    loadingText: COPY.modalities.pose.loadingText,
    missingLabel: COPY.modalities.pose.missingLabel,
    missingHint: COPY.modalities.pose.missingHint,
    placeholderIcon: "🧍",
    thumbnailSource: "overlay",
    storageKey: "pose",
    createExtractor: createPoseExtractor,
  },
  face: {
    title: COPY.modalities.face.trainerTitle,
    loadingText: COPY.modalities.face.loadingText,
    missingLabel: COPY.modalities.face.missingLabel,
    missingHint: COPY.modalities.face.missingHint,
    placeholderIcon: "😀",
    thumbnailSource: "overlay",
    storageKey: "face",
    createExtractor: createFaceExtractor,
  },
  images: {
    title: COPY.modalities.images.trainerTitle,
    loadingText: COPY.modalities.images.loadingText,
    missingLabel: COPY.modalities.images.missingLabel,
    missingHint: COPY.modalities.images.missingHint,
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

  const models = (["hands", "face", "pose", "images", "text", "audio"] as const).map((id) => ({
    id,
    title: COPY.modalities[id].cardTitle,
    description: COPY.modalities[id].cardDescription,
    enabled: true,
    cover: coverUrl(id),
  }));

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
            <img
              className="trainer-select-logo"
              src={`${import.meta.env.BASE_URL ?? "/"}brand/smartteam-logo.svg`}
              alt="SmartTEAM"
            />
            <h1 className="trainer-select-title">{COPY.selectTitle}</h1>
            <p className="trainer-select-subtitle">{COPY.selectSubtitle}</p>
          </div>
          <div className="trainer-select-header-right">
            <LangSwitch />
            {TURBOWARP_ENABLED && room && (
              <div className="trainer-select-room">Room: {room}</div>
            )}
          </div>
        </header>
        {TURBOWARP_ENABLED && !room && (
          <div className="trainer-select-warning">{COPY.selectNoSession}</div>
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
              {COPY.backToLobby}
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
