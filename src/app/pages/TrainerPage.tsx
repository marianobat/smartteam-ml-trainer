import { useState } from "react";
import HandTrainer from "./HandTrainer";
import { getToken, getRoom } from "../../core/bridge/session";
import "./TrainerPage.css";

const getRoomFromQuery = () => {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  return params.get("room") ?? "";
};

export default function TrainerPage() {
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const room = getRoomFromQuery() || getRoom() || "";
  const publishToken = getToken() || "";
  const [selectedModel, setSelectedModel] = useState<"hands" | null>(null);

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
      enabled: false,
      imageLabel: "Cara",
    },
    {
      id: "images",
      title: "Imagenes",
      description: "Reconocer objetos o escenas.",
      enabled: false,
      imageLabel: "Imagen",
    },
    {
      id: "pose",
      title: "Postura del cuerpo",
      description: "Pose completa con articulaciones.",
      enabled: false,
      imageLabel: "Cuerpo",
    },
    {
      id: "text",
      title: "Textos",
      description: "Clasificacion y comandos por texto.",
      enabled: false,
      imageLabel: "Texto",
    },
  ] as const;

  if (selectedModel !== "hands") {
    return (
      <div className="trainer-select">
        <header className="trainer-select-header">
          <div>
            <div className="trainer-select-kicker">SmartTEAM IA</div>
            <h1 className="trainer-select-title">Selecciona un modelo</h1>
            <p className="trainer-select-subtitle">
              Por ahora solo esta habilitado el modelo de gesto de las manos.
            </p>
          </div>
          <div className="trainer-select-room">Room: {room || "—"}</div>
        </header>
        {!room && (
          <div className="trainer-select-warning">
            No hay room disponible. Volve al lobby para crear una sesion.
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
                    setSelectedModel("hands");
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
    <HandTrainer
      onBack={() => window.location.assign(baseUrl)}
      room={room}
      publishToken={publishToken}
    />
  );
}
