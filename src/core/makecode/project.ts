// src/core/makecode/project.ts
//
// Arma el proyecto MakeCode que el shell le inyecta al fork vía el mensaje
// `importproject` (ver core/makecode/controller.ts). El proyecto trae la
// extensión BLE inline (como archivo del proyecto, no como dependencia de
// GitHub) y un main.ts generado con las clases reales, que el editor decompila
// a bloques al abrir.
//
// Inline en vez de `github:...`: el editor está deployado como build estático
// (pxt staticpkg) sin backend, así que el proxy /api/gh devuelve 404 y una
// dependencia de GitHub nunca resolvería. Embebiendo la fuente en el proyecto
// evitamos toda dependencia de red y mantenemos el fork mínimo.

import { generateBlocksCode, generateBlocksXml, type BlocksTransport } from "./codegen";
import bleExtensionSource from "./extensions/smartteam-ml-bluetooth.ts.txt?raw";

/** Mapa nombre-de-archivo → contenido (formato pxt.workspace.Project.text). */
export type ProjectText = Record<string, string>;

export interface MakeCodeProject {
  text: ProjectText;
}

export interface BuildProjectOptions {
  name?: string;
  transport?: BlocksTransport;
  classes: string[];
  includeNone?: boolean;
}

/** Archivo del proyecto con la fuente inline de la extensión BLE. */
const BLE_EXTENSION_FILE = "smartteamMLBT.ts";

/**
 * Construye el `project.text` para `importproject`. Generamos main.blocks (XML
 * de Blockly) con los bloques ya armados y main.ts equivalente: el editor en
 * modo controller NO decompila TS→bloques al importar, así que el XML es lo que
 * realmente se ve en el lienzo. La extensión BLE viaja inline como un archivo
 * más del proyecto, con `bluetooth` (paquete built-in) como dependencia y el
 * yotta config que habilita BLE.
 */
export function buildMakeCodeProject(options: BuildProjectOptions): MakeCodeProject {
  const transport = options.transport ?? "bluetooth";
  const includeNone = options.includeNone ?? true;
  const name = options.name ?? "SmartTEAM ML";

  const meta = { transport, classes: options.classes, includeNone };
  const mainTs = generateBlocksCode(meta);
  const mainBlocks = generateBlocksXml(meta);

  const config = {
    name,
    dependencies: {
      core: "*",
      bluetooth: "*",
    },
    yotta: {
      config: {
        "microbit-dal": {
          bluetooth: {
            open: 1,
          },
        },
      },
    },
    files: ["main.blocks", "main.ts", BLE_EXTENSION_FILE],
    preferredEditor: "blocksprj",
  };

  return {
    text: {
      "pxt.json": JSON.stringify(config, null, 4),
      "main.blocks": mainBlocks,
      "main.ts": mainTs,
      [BLE_EXTENSION_FILE]: bleExtensionSource,
    },
  };
}
