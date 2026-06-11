// src/core/microbit/protocol.ts
//
// Protocolo navegador ↔ micro:bit (texto por líneas, 115200 baudios):
//
//   Modo "a pedido" (recomendado): el micro:bit pregunta con "ML?\n" y el
//   navegador responde UNA línea "ML:<etiqueta>\n". Nunca llega un byte que
//   el micro:bit no pidió → no se llena el buffer RX.
//
//   Modo "automático" (compatibilidad con programas viejos): el navegador
//   empuja "ML:<etiqueta>\n" al cambiar la etiqueta y como heartbeat cada
//   RESEND_INTERVAL_MS.

export const BAUD_RATE = 115200;
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.7;
export const RESEND_INTERVAL_MS = 500;

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
