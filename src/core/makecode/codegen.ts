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

/**
 * Prefijo de los blockId de cada extensión (definidos con `//% blockId=...`).
 * Se usan para armar el XML de main.blocks que el editor renderiza directo
 * (importproject NO decompila TS→bloques en modo controller, así que el XML es
 * la única vía confiable para pre-armar los bloques).
 */
const BLOCK_ID_PREFIX: Record<BlocksTransport, string> = {
  usb: "smartteam_ml",
  bluetooth: "smartteam_mlbt",
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

/** Escapa texto para insertarlo en un atributo/nodo XML de Blockly. */
function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Clases válidas (sin vacías ni "none", que tiene su propio bloque). */
function detectableClasses(meta: BlocksMetadata): string[] {
  return meta.classes
    .map((c) => c.trim())
    .filter((c) => c.length > 0 && c.toLowerCase() !== NONE_LABEL);
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
  const classes = detectableClasses(meta);

  const lines: string[] = [];
  for (const name of classes) {
    lines.push(handlerBlock(`${ns}.alDetectarClase(${toStringLiteral(name)}, function ()`));
  }
  if (meta.includeNone) {
    lines.push(handlerBlock(`${ns}.cuandoNoHayDeteccion(function ()`));
  }

  return lines.length > 0 ? lines.join("\n") + "\n" : "";
}

/**
 * Construye el XML de `main.blocks`: un bloque-evento por clase ("al detectar
 * clase ML <nombre>") con el nombre real en un shadow de texto, más el bloque
 * "cuando no se detecta ninguna clase ML". El editor lo renderiza tal cual al
 * importar, con los cuerpos vacíos listos para que el chico arrastre acciones.
 */
export function generateBlocksXml(meta: BlocksMetadata): string {
  const prefix = BLOCK_ID_PREFIX[meta.transport];
  const classes = detectableClasses(meta);

  const X = 16;
  const STEP_Y = 140;
  const blocks: string[] = [];
  let y = 16;

  for (const name of classes) {
    blocks.push(
      [
        `  <block type="${prefix}_al_detectar" x="${X}" y="${y}">`,
        `    <value name="nombre"><shadow type="text"><field name="TEXT">${escapeXml(name)}</field></shadow></value>`,
        `    <statement name="HANDLER"></statement>`,
        `  </block>`,
      ].join("\n")
    );
    y += STEP_Y;
  }

  if (meta.includeNone) {
    blocks.push(
      [
        `  <block type="${prefix}_sin_deteccion" x="${X}" y="${y}">`,
        `    <statement name="HANDLER"></statement>`,
        `  </block>`,
      ].join("\n")
    );
  }

  const inner = blocks.length > 0 ? `\n${blocks.join("\n")}\n` : "";
  return `<xml xmlns="https://developers.google.com/blockly/xml">${inner}</xml>`;
}
