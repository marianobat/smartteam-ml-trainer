// src/core/microbit/serialConnection.ts
//
// Conexión Web Serial al micro:bit (Chrome/Edge). El micro:bit debe tener
// cargado un programa MakeCode con la extensión SmartTEAM ML, que pide la
// clase actual enviando líneas "ML?" por USB.

import { BAUD_RATE, formatLabelMessage, REQUEST_MESSAGE } from "./protocol";

export function isWebSerialSupported(): boolean {
  return typeof navigator !== "undefined" && "serial" in navigator;
}

let port: SerialPort | null = null;
let writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
let readLoopDone: Promise<void> | null = null;
let disconnectInFlight: Promise<void> | null = null;
let requestListener: (() => void) | null = null;
const encoder = new TextEncoder();

/**
 * Registra el callback que se dispara cada vez que el micro:bit envía "ML?".
 * Pasar null para dejar de escuchar.
 */
export function setMicrobitRequestListener(listener: (() => void) | null): void {
  requestListener = listener;
}

// Lee las líneas que manda el micro:bit. Solo reaccionamos a REQUEST_MESSAGE;
// cualquier otra salida serial del programa del chico se ignora.
async function readLoop(activeReader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";
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
  }
}

export function isMicrobitConnected(): boolean {
  return writer !== null;
}

export async function connectMicrobit(): Promise<void> {
  if (!isWebSerialSupported()) {
    throw new Error("Web Serial no está disponible en este navegador. Usá Chrome o Edge.");
  }
  await disconnectMicrobit();

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

  if (!selected.writable || !selected.readable) {
    await selected.close();
    throw new Error("El puerto seleccionado no permite lectura/escritura.");
  }

  port = selected;
  writer = selected.writable.getWriter();
  reader = selected.readable.getReader();
  readLoopDone = readLoop(reader);
}

export function disconnectMicrobit(): Promise<void> {
  if (!disconnectInFlight) {
    disconnectInFlight = doDisconnect().finally(() => {
      disconnectInFlight = null;
    });
  }
  return disconnectInFlight;
}

// Orden importante para que port.close() no quede esperando:
// 1) cancelar el reader y esperar a que el read loop suelte su lock,
// 2) abortar el writer (suelta el stream de escritura),
// 3) recién entonces cerrar el puerto.
async function doDisconnect(): Promise<void> {
  const activeReader = reader;
  reader = null;
  if (activeReader) {
    try {
      await activeReader.cancel();
    } catch {
      // el stream pudo haberse cerrado solo (p. ej. cable desenchufado)
    }
  }
  if (readLoopDone) {
    try {
      await readLoopDone;
    } catch {
      // idem
    }
    readLoopDone = null;
  }

  const activeWriter = writer;
  writer = null;
  if (activeWriter) {
    try {
      await activeWriter.abort();
    } catch {
      // el stream pudo haberse cerrado solo (p. ej. cable desenchufado)
    }
    // abort() NO libera el lock: sin esto, port.close() rechaza con
    // "Cannot close a locked stream" y el SO deja el puerto tomado,
    // bloqueando reconexiones desde MakeCode.
    try {
      activeWriter.releaseLock();
    } catch {
      // ya liberado
    }
  }

  const activePort = port;
  port = null;
  if (activePort) {
    try {
      await activePort.close();
    } catch (err) {
      console.warn("[microbit] port.close() falló; el puerto puede quedar tomado:", err);
    }
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
