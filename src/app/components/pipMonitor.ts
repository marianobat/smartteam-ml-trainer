// src/app/components/pipMonitor.ts
//
// Ventana flotante de monitoreo (Document Picture-in-Picture, Chrome 116+).
// Queda always-on-top sobre MakeCode y muestra: la cámara, la clase detectada
// con su confianza, y el estado de conexión del micro:bit. Comparte el
// contexto JS de la pestaña, así que lee los refs del Trainer en vivo —
// el modelo, la cámara y la conexión serial siguen viviendo en la página.

import { isMicrobitConnected } from "../../core/microbit/serialConnection";

export function isPipSupported(): boolean {
  return typeof window !== "undefined" && "documentPictureInPicture" in window;
}

export type PipMonitorOptions = {
  /** Video de la cámara del Trainer (se espeja en un canvas). */
  video: HTMLVideoElement | null;
  title: string;
  getLabel: () => string;
  getConfidence: () => number;
  /** Hay sujeto detectado en este momento. */
  isDetecting: () => boolean;
  /** Etiqueta para mostrar cuando no hay detección, p. ej. "Sin manos". */
  missingLabel: string;
  /** Umbral para colorear la predicción como aceptada. */
  acceptThreshold: number;
  /** Avisar al Trainer cuando la ventana se cierra (botón nativo o close()). */
  onClose: () => void;
};

/** Abre la ventana de monitoreo y devuelve una función para cerrarla. */
export async function openPipMonitor(opts: PipMonitorOptions): Promise<() => void> {
  const api = window.documentPictureInPicture;
  if (!api) {
    throw new Error("Este navegador no soporta la ventana de monitoreo (Chrome 116+).");
  }

  const pip = await api.requestWindow({ width: 360, height: 420 });
  const doc = pip.document;
  doc.title = `Monitoreo — ${opts.title}`;
  doc.body.style.cssText =
    "margin:0;padding:10px;background:#111;color:#fff;font-family:system-ui,sans-serif;" +
    "display:grid;gap:10px;align-content:start;box-sizing:border-box;";

  const titleEl = doc.createElement("div");
  titleEl.textContent = opts.title;
  titleEl.style.cssText = "font-size:12px;opacity:0.7;";
  doc.body.appendChild(titleEl);

  const canvas = doc.createElement("canvas");
  canvas.width = 320;
  canvas.height = 240;
  canvas.style.cssText =
    "width:100%;border-radius:10px;background:#000;transform:scaleX(-1);display:block;";
  doc.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const labelEl = doc.createElement("div");
  labelEl.style.cssText =
    "font-size:28px;font-weight:700;text-align:center;line-height:1.2;min-height:34px;";
  doc.body.appendChild(labelEl);

  const barWrap = doc.createElement("div");
  barWrap.style.cssText =
    "height:10px;background:#333;border-radius:999px;overflow:hidden;";
  const barFill = doc.createElement("div");
  barFill.style.cssText =
    "height:100%;width:0%;background:#22c55e;border-radius:999px;transition:width 120ms ease;";
  barWrap.appendChild(barFill);
  doc.body.appendChild(barWrap);

  const microbitEl = doc.createElement("div");
  microbitEl.style.cssText = "font-size:13px;display:flex;align-items:center;gap:6px;";
  doc.body.appendChild(microbitEl);

  let raf = 0;
  let closed = false;

  const tick = () => {
    if (closed) return;

    const video = opts.video;
    if (ctx && video && video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    const detecting = opts.isDetecting();
    const label = detecting ? opts.getLabel() || "—" : opts.missingLabel;
    const confidence = detecting ? opts.getConfidence() : 0;
    const accepted = detecting && confidence >= opts.acceptThreshold;

    labelEl.textContent = confidence > 0 ? `${label} (${confidence.toFixed(2)})` : label;
    labelEl.style.color = accepted ? "#22c55e" : detecting ? "#e5e5e5" : "#888";
    barFill.style.width = `${Math.round(Math.max(0, Math.min(1, confidence)) * 100)}%`;
    barFill.style.background = accepted ? "#22c55e" : "#666";

    const connected = isMicrobitConnected();
    microbitEl.textContent = connected ? "● micro:bit conectado" : "○ micro:bit desconectado";
    microbitEl.style.color = connected ? "#22c55e" : "#888";

    raf = pip.requestAnimationFrame(tick);
  };
  raf = pip.requestAnimationFrame(tick);

  const cleanup = () => {
    if (closed) return;
    closed = true;
    pip.cancelAnimationFrame(raf);
    opts.onClose();
  };

  pip.addEventListener("pagehide", cleanup);

  return () => {
    cleanup();
    try {
      pip.close();
    } catch {
      // la ventana ya estaba cerrada
    }
  };
}
