// src/core/microbit/transport.ts
//
// Capa común a los dos transportes del micro:bit (Web Serial y Web Bluetooth):
// listeners compartidos y parser de líneas entrantes. Hay a lo sumo UNA
// conexión activa a la vez (de cualquier transporte), así que los listeners
// son globales como antes.

import { ALIAS_PREFIX, REQUEST_MESSAGE, sanitizeAlias } from "./protocol";

export type MicrobitTransportKind = "serial" | "bluetooth";

export type MicrobitListeners = {
  /** El micro:bit pidió la clase actual ("ML?"). */
  onRequest: (() => void) | null;
  /** El programa MakeCode nombró la placa ("ML@<alias>"). */
  onAlias: ((alias: string) => void) | null;
  /** La conexión se cayó sola (cable/BLE), no por pedido del usuario. */
  onDrop: (() => void) | null;
};

const listeners: MicrobitListeners = { onRequest: null, onAlias: null, onDrop: null };

export function setMicrobitListeners(next: Partial<MicrobitListeners>): void {
  Object.assign(listeners, next);
}

export function clearMicrobitListeners(): void {
  listeners.onRequest = null;
  listeners.onAlias = null;
  listeners.onDrop = null;
}

export function notifyDrop(): void {
  listeners.onDrop?.();
}

export function handleIncomingLine(line: string): void {
  if (line === REQUEST_MESSAGE) {
    listeners.onRequest?.();
    return;
  }
  if (line.startsWith(ALIAS_PREFIX)) {
    const alias = sanitizeAlias(line.slice(ALIAS_PREFIX.length));
    if (alias) listeners.onAlias?.(alias);
  }
  // cualquier otra salida del programa del chico se ignora
}

/** Acumula chunks de texto y despacha línea por línea a handleIncomingLine. */
export function createLineBuffer() {
  let buffer = "";
  return {
    push(chunk: string) {
      buffer += chunk;
      let newlineIdx = buffer.indexOf("\n");
      while (newlineIdx >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (line) handleIncomingLine(line);
        newlineIdx = buffer.indexOf("\n");
      }
      // si el programa imprime sin saltos de línea, no acumular sin límite
      if (buffer.length > 256) buffer = buffer.slice(-256);
    },
  };
}
