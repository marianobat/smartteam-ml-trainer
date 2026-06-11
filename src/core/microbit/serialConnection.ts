// src/core/microbit/serialConnection.ts
//
// Conexión Web Serial al micro:bit (Chrome/Edge). El micro:bit debe tener
// cargado un programa MakeCode con la extensión SmartTEAM ML, que lee las
// líneas "ML:<etiqueta>" por USB.

import { BAUD_RATE, formatLabelMessage, REQUEST_MESSAGE } from "./protocol";

export function isWebSerialSupported(): boolean {
  return typeof navigator !== "undefined" && "serial" in navigator;
}

let port: SerialPort | null = null;
let writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
let requestListener: (() => void) | null = null;
const encoder = new TextEncoder();

/**
 * Registra el callback que se dispara cada vez que el micro:bit envía "ML?"
 * (modo "a pedido"). Pasar null para dejar de escuchar.
 */
export function setMicrobitRequestListener(listener: (() => void) | null): void {
  requestListener = listener;
}

// Lee las líneas que manda el micro:bit. Solo reaccionamos a REQUEST_MESSAGE;
// cualquier otra salida serial del programa del chico se ignora.
async function startReadLoop(activePort: SerialPort): Promise<void> {
  if (!activePort.readable) return;
  const decoder = new TextDecoder();
  let buffer = "";
  reader = activePort.readable.getReader();
  const activeReader = reader;
  try {
    for (;;) {
      const { value, done } = await activeReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIdx = buffer.indexOf("\n");
      while (newlineIdx >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (line === REQUEST_MESSAGE) {
          requestListener?.();
        }
        newlineIdx = buffer.indexOf("\n");
      }
      // si el programa del chico imprime sin saltos de línea, no acumular sin límite
      if (buffer.length > 256) buffer = buffer.slice(-256);
    }
  } catch {
    // puerto cerrado o cable desenchufado: el loop termina solo
  } finally {
    try {
      activeReader.releaseLock();
    } catch {
      // ya liberado
    }
    if (reader === activeReader) reader = null;
  }
}

export function isMicrobitConnected(): boolean {
  return writer !== null;
}

export async function connectMicrobit(): Promise<void> {
  if (!isWebSerialSupported()) {
    throw new Error("Web Serial no está disponible en este navegador. Usá Chrome o Edge.");
  }
  if (port) {
    await disconnectMicrobit();
  }

  // Abre el diálogo del navegador para elegir el puerto del micro:bit
  const selected = await navigator.serial!.requestPort();

  try {
    await selected.open({ baudRate: BAUD_RATE });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (/open|in use|busy|failed/i.test(message)) {
      throw new Error(
        "No se pudo abrir el puerto. Si MakeCode está conectado al micro:bit en otra pestaña, desconectalo ahí (o cerrá esa pestaña) y probá de nuevo."
      );
    }
    throw err;
  }

  if (!selected.writable) {
    await selected.close();
    throw new Error("El puerto seleccionado no permite escritura.");
  }

  port = selected;
  writer = selected.writable.getWriter();
  void startReadLoop(selected);
}

export async function disconnectMicrobit(): Promise<void> {
  if (reader) {
    try {
      await reader.cancel();
    } catch {
      // el stream pudo haberse cerrado solo
    }
    reader = null;
  }
  if (writer) {
    try {
      writer.releaseLock();
    } catch {
      // el stream pudo haberse cerrado solo (p. ej. cable desenchufado)
    }
    writer = null;
  }
  if (port) {
    try {
      await port.close();
    } catch {
      // idem
    }
    port = null;
  }
}

/** Envía "ML:<etiqueta>\n" y devuelve la línea enviada (sin el salto). */
export async function sendMicrobitLabel(label: string): Promise<string> {
  if (!writer) {
    throw new Error("micro:bit no conectado.");
  }
  const line = formatLabelMessage(label);
  try {
    await writer.write(encoder.encode(line));
  } catch (err) {
    // Si la escritura falla (cable desenchufado), dejamos la conexión limpia
    await disconnectMicrobit();
    throw err;
  }
  return line.trimEnd();
}
