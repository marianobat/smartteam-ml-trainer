// src/core/extractors/types.ts

/** Identificador de modalidad de captura por video. */
export type VideoModality = "hands" | "pose" | "face" | "image";

/**
 * Contrato común de extractores de features sobre frames de video.
 * Agregar una modalidad nueva = implementar este contrato; el resto
 * (dataset, entrenamiento, predicción, publicación WS) es genérico.
 */
export type VideoExtractor = {
  id: VideoModality;
  /** Largo del vector de features que produce processFrame. */
  featureDim: number;
  /**
   * Intervalo mínimo entre llamadas a processFrame (ms). Para extractores
   * pesados (p. ej. MobileNet) evita correr la inferencia en cada frame.
   * Sin definir = procesar todos los frames.
   */
  frameIntervalMs?: number;
  /** Descarga e inicializa el modelo base (WASM + .task la primera vez). */
  load(): Promise<void>;
  /**
   * Detecta sobre el frame actual, dibuja el overlay en el canvas y
   * devuelve el vector de features (null si no hay sujeto detectado).
   */
  processFrame(
    video: HTMLVideoElement,
    ctx: CanvasRenderingContext2D,
    timestampMs: number
  ): Float32Array | null;
};
