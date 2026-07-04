// src/app/components/trainer/CaptureControls.tsx
//
// Controles de captura superpuestos sobre la cámara: toggle "de a una /
// ráfaga" + botón redondo grande verde (mantener apretado para ráfaga).

import type { MouseEvent, TouchEvent } from "react";
import { Camera, Layers } from "lucide-react";
import { COPY } from "../../copy";
import "./CaptureControls.css";

type PressEvent = MouseEvent<HTMLButtonElement> | TouchEvent<HTMLButtonElement>;

type CaptureControlsProps = {
  disabled: boolean;
  burstMode: boolean;
  onToggleBurst: () => void;
  onPressStart: (event: PressEvent) => void;
  onPressEnd: (event: PressEvent) => void;
};

export default function CaptureControls({
  disabled,
  burstMode,
  onToggleBurst,
  onPressStart,
  onPressEnd,
}: CaptureControlsProps) {
  return (
    <div className="capture-controls">
      <div className="capture-mode" role="group" aria-label="Modo de captura">
        <button
          type="button"
          className={`capture-mode-btn ${!burstMode ? "is-on" : ""}`}
          aria-pressed={!burstMode}
          onClick={() => burstMode && onToggleBurst()}
          title={COPY.captureOne}
          aria-label={COPY.captureOne}
        >
          <Camera size={20} aria-hidden="true" />
        </button>
        <button
          type="button"
          className={`capture-mode-btn ${burstMode ? "is-on" : ""}`}
          aria-pressed={burstMode}
          onClick={() => !burstMode && onToggleBurst()}
          title={`${COPY.captureBurst} — ${COPY.captureHint}`}
          aria-label={COPY.captureBurst}
        >
          <Layers size={20} aria-hidden="true" />
        </button>
      </div>

      <button
        type="button"
        className="capture-shutter"
        aria-label="Capturar ejemplo"
        disabled={disabled}
        onMouseDown={onPressStart}
        onMouseUp={onPressEnd}
        onMouseLeave={onPressEnd}
        onTouchStart={onPressStart}
        onTouchEnd={onPressEnd}
        onTouchCancel={onPressEnd}
      >
        <span className="capture-shutter-dot" aria-hidden="true" />
      </button>
    </div>
  );
}
