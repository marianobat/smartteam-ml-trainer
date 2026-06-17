// src/core/makecode/project.ts
//
// Arma el proyecto MakeCode que el shell le inyecta al fork vía el mensaje
// `importproject` (ver core/makecode/controller.ts). El proyecto trae la
// extensión BLE como dependencia y un main.ts generado con las clases reales,
// que el editor decompila a bloques al abrir.

import { MAKECODE_BLE_DEP } from "../bridge/config";
import { generateBlocksCode, type BlocksTransport } from "./codegen";

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
  /** Dependencia de la extensión de bloques (github:owner/repo[#ref]). */
  bleDependency?: string;
}

const EMPTY_BLOCKS = '<xml xmlns="https://developers.google.com/blockly/xml"></xml>';

/**
 * Construye el `project.text` para `importproject`. Incluimos main.blocks vacío
 * + main.ts con el código generado: al importar con TS no vacío y bloques
 * vacíos, el editor decompila el TS y muestra los bloques ya armados.
 */
export function buildMakeCodeProject(options: BuildProjectOptions): MakeCodeProject {
  const transport = options.transport ?? "bluetooth";
  const includeNone = options.includeNone ?? true;
  const name = options.name ?? "SmartTEAM ML";
  const dependency = options.bleDependency ?? MAKECODE_BLE_DEP;

  const mainTs = generateBlocksCode({ transport, classes: options.classes, includeNone });

  const config = {
    name,
    dependencies: {
      core: "*",
      "smartteam-ml-bluetooth": dependency,
    },
    files: ["main.blocks", "main.ts"],
    preferredEditor: "blocksprj",
  };

  return {
    text: {
      "pxt.json": JSON.stringify(config, null, 4),
      "main.blocks": EMPTY_BLOCKS,
      "main.ts": mainTs,
    },
  };
}
