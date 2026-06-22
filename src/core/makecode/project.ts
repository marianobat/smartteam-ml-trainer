// src/core/makecode/project.ts
//
// Arma el proyecto MakeCode que el shell le inyecta al fork vía el mensaje
// `importproject` (ver core/makecode/controller.ts). El canvas arranca vacío
// (sin bloques pre-armados) pero con la extensión BLE inline en el toolbox y un
// archivo `clases.ts` generado que expone las clases entrenadas como desplegable.
//
// Inline en vez de `github:...`: el editor está deployado como build estático
// (pxt staticpkg) sin backend, así que el proxy /api/gh devuelve 404 y una
// dependencia de GitHub nunca resolvería. Embebiendo la fuente en el proyecto
// evitamos toda dependencia de red y mantenemos el fork mínimo.

import { generateClassesFile, type BlocksTransport } from "./codegen";
import bleExtensionSource from "./extensions/smartteam-ml-bluetooth.ts.txt?raw";

/** Mapa nombre-de-archivo → contenido (formato pxt.workspace.Project.text). */
export type ProjectText = Record<string, string>;

export interface MakeCodeProject {
  text: ProjectText;
}

export interface BuildProjectOptions {
  name?: string;
  transport?: BlocksTransport;
  /** Clases del modelo entrenado → se exponen como desplegable en `clases.ts`. */
  classes?: string[];
}

/** Archivo del proyecto con la fuente inline de la extensión BLE. */
const BLE_EXTENSION_FILE = "smartteamMLBT.ts";
/** Archivo generado con el enum de clases (desplegable) y sus bloques. */
const CLASSES_FILE = "clases.ts";

const EMPTY_BLOCKS = '<xml xmlns="https://developers.google.com/blockly/xml"></xml>';

/**
 * Construye el `project.text` para `importproject`. El canvas arranca vacío (sin
 * bloques pre-armados); trae la extensión BLE inline más `clases.ts`, que define
 * el enum con las clases entrenadas y los bloques de desplegable. La extensión
 * viaja como un archivo más del proyecto, con `bluetooth` (paquete built-in)
 * como dependencia y el yotta config que habilita BLE.
 */
export function buildMakeCodeProject(options: BuildProjectOptions = {}): MakeCodeProject {
  const name = options.name ?? "SmartTEAM ML";
  const transport = options.transport ?? "bluetooth";
  const classesSource = generateClassesFile({ transport, classes: options.classes ?? [] });

  const config = {
    name,
    dependencies: {
      core: "*",
      bluetooth: "*",
    },
    // "No Pairing Required": la placa advierte y acepta conexión SIN emparejar,
    // que es lo que necesita el Web Bluetooth del trainer. Sin esto, el modo por
    // defecto (JustWorks) exige emparejamiento y el navegador no logra conectar.
    // open=1 (abierto), whitelist=0 (cualquiera puede conectar). Vale V1 y V2.
    yotta: {
      config: {
        "microbit-dal": {
          bluetooth: {
            enabled: 1,
            open: 1,
            whitelist: 0,
          },
        },
      },
    },
    files: ["main.blocks", "main.ts", BLE_EXTENSION_FILE, CLASSES_FILE],
    preferredEditor: "blocksprj",
  };

  return {
    text: {
      "pxt.json": JSON.stringify(config, null, 4),
      "main.blocks": EMPTY_BLOCKS,
      "main.ts": "",
      [BLE_EXTENSION_FILE]: bleExtensionSource,
      [CLASSES_FILE]: classesSource,
    },
  };
}
