// src/core/microbit/protocol.ts
//
// Protocolo navegador → micro:bit: una etiqueta por línea, "ML:<etiqueta>\n".
// Se envía cuando cambia la etiqueta ganadora o cada RESEND_INTERVAL_MS como
// heartbeat. La extensión MakeCode (smartteam-makecode-extension) parsea estas
// líneas con serial.onDataReceived.

export const BAUD_RATE = 115200;
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.7;
export const RESEND_INTERVAL_MS = 500;

/** Etiqueta que indica "no hay detección" (sin sujeto o confianza baja). */
export const NONE_LABEL = "none";

/** La etiqueta viaja en una línea de texto: sin saltos de línea ni ":". */
export function sanitizeLabel(label: string): string {
  return label.replace(/[\r\n:]+/g, " ").trim() || NONE_LABEL;
}

export function formatLabelMessage(label: string): string {
  return `ML:${sanitizeLabel(label)}\n`;
}
