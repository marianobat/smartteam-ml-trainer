// src/core/makecode/codegen.ts
//
// Genera el archivo `clases.ts` que se inyecta en el proyecto del chico vía
// importproject (postMessage al fork de MakeCode). Define un `enum ClaseML` con
// las clases reales del modelo entrenado (dropdown nativo de MakeCode) y unos
// bloques que mapean ese enum a la API de string de la extensión BLE inline.
//
// Así el bloque "al detectar clase ML <clase>" muestra un desplegable con las
// clases entrenadas, sin tocar el fork ni depender de la red.

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

/** Prefijo de los blockId de cada extensión (para IDs únicos y estables). */
const BLOCK_ID_PREFIX: Record<BlocksTransport, string> = {
  usb: "smartteam_ml",
  bluetooth: "smartteam_mlbt",
};

export interface ClassesMetadata {
  /** Transporte → define el namespace de los bloques generados. */
  transport: BlocksTransport;
  /** Nombres canónicos de las clases entrenadas, en orden. */
  classes: string[];
}

/** Clases válidas (sin vacías ni "none", que tiene su propio bloque aparte). */
function detectableClasses(classes: string[]): string[] {
  return classes
    .map((c) => c.trim())
    .filter((c) => c.length > 0 && c.toLowerCase() !== NONE_LABEL);
}

/** Escapa un nombre para usarlo como literal de string en TS. */
function toStringLiteral(name: string): string {
  const escaped = name.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  return `"${escaped}"`;
}

/** Etiqueta para `//% block="..."`: una sola línea, sin comillas dobles. */
function toBlockLabel(name: string): string {
  return name.replace(/\s+/g, " ").replace(/"/g, "'").trim();
}

/**
 * Genera el contenido de `clases.ts`: el enum con las clases reales (dropdown) y
 * los bloques `al detectar clase ML %clase` / `clase ML es %clase` que lo mapean
 * a la API de string de la extensión. Si no hay clases (modelo sin entrenar),
 * usa una entrada placeholder para que el bloque exista igual.
 */
export function generateClassesFile(meta: ClassesMetadata): string {
  const ns = NAMESPACE[meta.transport];
  const prefix = BLOCK_ID_PREFIX[meta.transport];
  const found = detectableClasses(meta.classes);
  const list = found.length > 0 ? found : ["clase 1"];

  const enumMembers = list
    .map((name, i) => `    //% block="${toBlockLabel(name)}"\n    Clase${i} = ${i}`)
    .join(",\n");

  const nombres = list.map(toStringLiteral).join(", ");

  return `// Generado por el trainer: clases del modelo entrenado como desplegable.
enum ClaseML {
${enumMembers}
}

namespace ${ns} {
    const NOMBRES_CLASE = [${nombres}]

    /**
     * Ejecuta el código cuando el entrenador detecta la clase elegida.
     */
    //% blockId=${prefix}_al_detectar_sel
    //% block="al detectar clase ML %clase"
    //% weight=99
    export function alDetectarClaseSel(clase: ClaseML, manejador: () => void): void {
        alDetectarClase(NOMBRES_CLASE[clase], manejador)
    }

    /**
     * Verdadero si la clase detectada en este momento es la elegida.
     */
    //% blockId=${prefix}_clase_es_sel
    //% block="clase ML es %clase"
    //% weight=85
    export function claseEsSel(clase: ClaseML): boolean {
        return claseEs(NOMBRES_CLASE[clase])
    }
}
`;
}
