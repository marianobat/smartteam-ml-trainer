// src/app/components/trainer/CameraStage.tsx
//
// Escenario de cámara: video espejado (atenuado en modalidades con esqueleto),
// canvas de overlay encima, hint cuando no hay detección y controles
// superpuestos (children) abajo al centro. Con `focusBox` (imágenes) el video
// se ve borroso salvo la ventana 4:3 central: la misma zona que el extractor
// recorta como muestra (ver core/extractors/focusBox.ts).

import { useEffect, useState, type CSSProperties, type ReactNode, type RefObject } from "react";
import { focusBoxRect } from "../../../core/extractors/focusBox";
import "./CameraStage.css";

type CameraStageProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  /** Atenuar el video para que el esqueleto sea protagonista (no en imágenes). */
  dimmed: boolean;
  /** Ventana 4:3 nítida al centro; el resto borroso (modalidad imágenes). */
  focusBox?: boolean;
  loading: boolean;
  loadingText: string;
  /** Mensaje cuando no se detecta el sujeto (null = no mostrar). */
  hint: string | null;
  /** Píldora superior (p. ej. "Veo: Abierta 98%" durante el paso Probar). */
  overlay?: ReactNode;
  children?: ReactNode;
};

/** Variables CSS --fb-* del recuadro, en % del frame (misma fuente que el extractor). */
function focusBoxCssVars(frameWidth: number, frameHeight: number): CSSProperties {
  const rect = focusBoxRect(frameWidth, frameHeight);
  const pct = (value: number, total: number) => `${(value / total) * 100}%`;
  return {
    "--fb-left": pct(rect.x, frameWidth),
    "--fb-top": pct(rect.y, frameHeight),
    "--fb-w": pct(rect.width, frameWidth),
    "--fb-h": pct(rect.height, frameHeight),
  } as CSSProperties;
}

export default function CameraStage({
  videoRef,
  canvasRef,
  dimmed,
  focusBox = false,
  loading,
  loadingText,
  hint,
  overlay,
  children,
}: CameraStageProps) {
  // Tamaño real del frame para alinear el recuadro con el recorte del extractor
  const [frame, setFrame] = useState<{ w: number; h: number }>({ w: 640, h: 480 });

  useEffect(() => {
    if (!focusBox) return;
    const video = videoRef.current;
    if (!video) return;
    const update = () => {
      if (video.videoWidth > 0) {
        setFrame({ w: video.videoWidth, h: video.videoHeight });
      }
    };
    update();
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [focusBox, videoRef]);

  return (
    <div className={`camera-stage ${dimmed ? "is-dimmed" : ""}`}>
      <video ref={videoRef} className="camera-stage-video" playsInline muted />
      <canvas ref={canvasRef} className="camera-stage-canvas" />
      {focusBox && !loading && (
        <div className="camera-stage-focus" style={focusBoxCssVars(frame.w, frame.h)} aria-hidden="true">
          <span className="camera-stage-focus-blur is-top" />
          <span className="camera-stage-focus-blur is-bottom" />
          <span className="camera-stage-focus-blur is-left" />
          <span className="camera-stage-focus-blur is-right" />
          <span className="camera-stage-focus-frame" />
        </div>
      )}
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
