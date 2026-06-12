// src/core/microbit/bluetoothConnection.ts
//
// Conexión Web Bluetooth al micro:bit (Chrome/Edge, también Android/ChromeOS)
// usando el servicio UART de Nordic (NUS). El micro:bit debe tener grabado un
// programa MakeCode con la extensión SmartTEAM ML Bluetooth (requiere V2).
//
// Mismo protocolo "a pedido" que USB: el micro:bit pregunta "ML?\n" y acá se
// responde "ML:<etiqueta>\n". Además puede anunciar un alias con "ML@<alias>".

import { formatLabelMessage } from "./protocol";
import { createLineBuffer, notifyDrop } from "./transport";

const NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
/** Escritura: navegador → micro:bit. */
const NUS_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";
/** Notificaciones: micro:bit → navegador. */
const NUS_TX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";

// BLE escribe de a paquetes chicos (MTU típico 23 → 20 bytes útiles)
const CHUNK_SIZE = 20;

let device: BluetoothDevice | null = null;
let rxChar: BluetoothRemoteGATTCharacteristic | null = null;
let connected = false;
const encoder = new TextEncoder();

export function isWebBluetoothSupported(): boolean {
  return typeof navigator !== "undefined" && "bluetooth" in navigator;
}

export function isMicrobitBleConnected(): boolean {
  return connected;
}

export function getBleDeviceName(): string {
  // "BBC micro:bit [zatig]" → "zatig" si se puede, nombre completo si no
  const name = device?.name ?? "";
  const match = name.match(/\[(.+)\]/);
  return match ? match[1] : name;
}

function handleUnexpectedDisconnect() {
  const wasConnected = connected;
  connected = false;
  rxChar = null;
  if (wasConnected) {
    notifyDrop();
  }
}

/** Abre el selector de dispositivos y conecta. Devuelve el nombre de la placa. */
export async function connectMicrobitBle(): Promise<string> {
  if (!isWebBluetoothSupported()) {
    throw new Error("Web Bluetooth no está disponible en este navegador. Usá Chrome o Edge.");
  }
  await disconnectMicrobitBle();

  const selected = await navigator.bluetooth!.requestDevice({
    filters: [{ namePrefix: "BBC micro:bit" }],
    optionalServices: [NUS_SERVICE],
  });

  const gatt = selected.gatt;
  if (!gatt) {
    throw new Error("El dispositivo elegido no soporta conexión GATT.");
  }

  const server = await gatt.connect();

  let service: BluetoothRemoteGATTService;
  try {
    service = await server.getPrimaryService(NUS_SERVICE);
  } catch {
    try {
      gatt.disconnect();
    } catch {
      // ya desconectado
    }
    throw new Error(
      "La placa no tiene el servicio Bluetooth UART. Grabale un programa MakeCode con la extensión SmartTEAM ML Bluetooth y probá de nuevo."
    );
  }

  try {
    const txChar = await service.getCharacteristic(NUS_TX);
    rxChar = await service.getCharacteristic(NUS_RX);

    const decoder = new TextDecoder();
    const lineBuffer = createLineBuffer();
    txChar.addEventListener("characteristicvaluechanged", (event) => {
      const value = (event.target as BluetoothRemoteGATTCharacteristic).value;
      if (value) {
        lineBuffer.push(decoder.decode(value));
      }
    });
    await txChar.startNotifications();
  } catch (err) {
    rxChar = null;
    try {
      gatt.disconnect();
    } catch {
      // ya desconectado
    }
    const message = err instanceof Error ? err.message : String(err);
    if (/authentication|insufficient|security/i.test(message)) {
      throw new Error(
        'La placa está exigiendo emparejamiento. En MakeCode: Configuración del proyecto → activá "No Pairing Required", volvé a descargar y regrabar el programa. Si la placa figura emparejada en el Bluetooth del sistema, eliminala de ahí también.'
      );
    }
    throw err;
  }

  selected.addEventListener("gattserverdisconnected", handleUnexpectedDisconnect);

  device = selected;
  connected = true;
  return getBleDeviceName() || "micro:bit";
}

export async function disconnectMicrobitBle(): Promise<void> {
  const activeDevice = device;
  device = null;
  rxChar = null;
  connected = false;
  if (activeDevice) {
    // desconexión pedida por el usuario: que no dispare onDrop
    activeDevice.removeEventListener("gattserverdisconnected", handleUnexpectedDisconnect);
    try {
      activeDevice.gatt?.disconnect();
    } catch {
      // ya desconectado
    }
  }
}

/** Envía "ML:<etiqueta>\n" troceado en paquetes BLE. Devuelve la línea enviada. */
export async function sendMicrobitLabelBle(label: string): Promise<string> {
  const char = rxChar;
  if (!char || !connected) {
    throw new Error("micro:bit no conectado por Bluetooth.");
  }
  const line = formatLabelMessage(label);
  const bytes = encoder.encode(line);
  try {
    for (let offset = 0; offset < bytes.length; offset += CHUNK_SIZE) {
      const chunk = bytes.slice(offset, offset + CHUNK_SIZE);
      if (char.properties?.writeWithoutResponse && char.writeValueWithoutResponse) {
        await char.writeValueWithoutResponse(chunk);
      } else {
        await char.writeValue(chunk);
      }
    }
  } catch (err) {
    // Si la escritura falla (placa apagada / fuera de alcance), limpiar
    await disconnectMicrobitBle();
    throw err;
  }
  return line.trimEnd();
}
