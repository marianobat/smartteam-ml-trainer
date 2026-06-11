// src/core/microbit/protocol.ts
//
// Protocolo navegador ↔ micro:bit (texto por líneas, 115200 baudios), siempre
// "a pedido": el micro:bit pregunta con "ML?\n" y el navegador responde UNA
// línea "ML:<etiqueta>\n". Nunca llega un byte que el micro:bit no pidió, así
// que su buffer RX no puede llenarse. El ritmo lo marca la extensión MakeCode
// (sondeo en segundo plano + bloque "pedir clase ML").

export const BAUD_RATE = 115200;
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.7;

/** Línea que envía el micro:bit para pedir la clase actual. */
export const REQUEST_MESSAGE = "ML?";

/** Etiqueta que indica "no hay detección" (sin sujeto o confianza baja). */
export const NONE_LABEL = "none";

/** La etiqueta viaja en una línea de texto: sin saltos de línea ni ":". */
export function sanitizeLabel(label: string): string {
  return label.replace(/[\r\n:]+/g, " ").trim() || NONE_LABEL;
}

export function formatLabelMessage(label: string): string {
  return `ML:${sanitizeLabel(label)}\n`;
}
