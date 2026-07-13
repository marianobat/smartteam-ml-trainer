// src/app/components/trainer/CameraStage.tsx
//
// Escenario de cámara: video espejado (atenuado en modalidades con esqueleto),
// canvas de overlay encima, hint cuando no hay detección y controles
// superpuestos (children) abajo al centro.

import type { ReactNode, RefObject } from "react";
import "./CameraStage.css";

type CameraStageProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  /** Atenuar el video para que el esqueleto sea protagonista (no en imágenes). */
  dimmed: boolean;
  loading: boolean;
  loadingText: string;
  /** Mensaje cuando no se detecta el sujeto (null = no mostrar). */
  hint: string | null;
  /** Píldora superior (p. ej. "Veo: Abierta 98%" durante el paso Probar). */
  overlay?: ReactNode;
  children?: ReactNode;
};

export default function CameraStage({
  videoRef,
  canvasRef,
  dimmed,
  loading,
  loadingText,
  hint,
  overlay,
  children,
}: CameraStageProps) {
  return (
    <div className={`camera-stage ${dimmed ? "is-dimmed" : ""}`}>
      <video ref={videoRef} className="camera-stage-video" playsInline muted />
      <canvas ref={canvasRef} className="camera-stage-canvas" />
      {loading && (
        <div className="camera-stage-loading" role="status">
          <span className="camera-stage-spinner" aria-hidden="true" />
          {loadingText}
        </div>
      )}
      {!loading && overlay && <div className="camera-stage-overlay">{overlay}</div>}
      {!loading && hint && <div className="camera-stage-hint">{hint}</div>}
      {!loading && <div className="camera-stage-controls">{children}</div>}
    </div>
  );
}
