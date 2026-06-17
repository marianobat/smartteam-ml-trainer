// src/core/makecode/codegen.ts
//
// Genera el TypeScript que se inyecta en el proyecto del chico vía
// importproject (postMessage al fork de MakeCode). El editor lo decompila a
// bloques, así que el resultado son bloques "al detectar clase ML <nombre>" YA
// armados con los nombres reales de las clases entrenadas (en vez de un
// dropdown dinámico forkeado).

const NONE_LABEL = "none";

/**
 * Namespace de la extensión de bloques según el transporte. Son extensiones
 * distintas a propósito (bluetooth es incompatible con radio en MakeCode):
 * - USB        → smartteam-makecode-extension  → namespace `smartteamML`
 * - Bluetooth  → smartteam-ml-bluetooth        → namespace `smartteamMLBT`
 */
export type BlocksTransport = "usb" | "bluetooth";

const NAMESPACE: Record<BlocksTransport, string> = {
  usb: "smartteamML",
  bluetooth: "smartteamMLBT",
};

export interface BlocksMetadata {
  /** Transporte → define el namespace de los bloques generados. */
  transport: BlocksTransport;
  /** Nombres canónicos de las clases, en orden. */
  classes: string[];
  /** Incluir el manejador "cuando no se detecta ninguna clase". */
  includeNone: boolean;
}

/** Escapa un nombre para usarlo como literal de string en TS. */
function toStringLiteral(name: string): string {
  const escaped = name.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  return `"${escaped}"`;
}

function handlerBlock(call: string): string {
  // El cuerpo vacío con una línea en blanco indentada deja el bloque listo
  // para que el chico arrastre acciones dentro.
  return `${call} {\n    \n})`;
}

/**
 * Construye el TS a inyectar a partir de la lista de clases.
 * Filtra "none" de los eventos por clase (tiene su propio bloque).
 */
export function generateBlocksCode(meta: BlocksMetadata): string {
  const ns = NAMESPACE[meta.transport];
  const classes = meta.classes
    .map((c) => c.trim())
    .filter((c) => c.length > 0 && c.toLowerCase() !== NONE_LABEL);

  const lines: string[] = [];
  for (const name of classes) {
    lines.push(handlerBlock(`${ns}.alDetectarClase(${toStringLiteral(name)}, function ()`));
  }
  if (meta.includeNone) {
    lines.push(handlerBlock(`${ns}.cuandoNoHayDeteccion(function ()`));
  }

  return lines.length > 0 ? lines.join("\n") + "\n" : "";
}
