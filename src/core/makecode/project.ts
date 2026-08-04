// src/core/makecode/project.ts
//
// Arma el proyecto MakeCode que el shell le inyecta al editor vía el mensaje
// `importproject` (ver core/makecode/controller.ts). El canvas arranca vacío
// (sin bloques pre-armados) pero con la extensión BLE inline en el toolbox, un
// archivo `clases.ts` generado que expone las clases entrenadas como desplegable
// y —si el curso tiene extensión publicada— esa extensión como dependencia.
//
// Dos mecanismos, a propósito distintos:
//  - Extensión BLE base (smartteamMLBT): INLINE, como archivo del proyecto. Es
//    un único .ts sin dependencias ni assets; embebida evita depender de red y
//    mantiene el flujo BLE andando incluso en un editor sin proxy /api/gh.
//  - Extensión de curso (ext3–ext9, multi-archivo + icono + locales): como
//    DEPENDENCIA de GitHub (`github:owner/repo#tag`). Inline no escala a esos
//    paquetes; la dependencia deja que el editor resuelva todo el paquete. Esto
//    asume un editor que resuelve GitHub (el oficial lo hace; ver courses.ts).

import { generateClassesFile, type BlocksTransport } from "./codegen";
import bleExtensionSource from "./extensions/smartteam-ml-bluetooth.ts.txt?raw";
import { COURSES, type CourseId, type CoursePackage } from "./courses";

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
  /** Curso (3.º–9.º): define qué extensión se agrega como dependencia (ver courses.ts). */
  course?: CourseId;
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
 * BLE viaja como un archivo más del proyecto, con `bluetooth` (paquete built-in)
 * como dependencia y el yotta config que habilita BLE. Si el curso tiene
 * extensión publicada, se suma a `dependencies` como paquete de GitHub.
 */
export function buildMakeCodeProject(options: BuildProjectOptions = {}): MakeCodeProject {
  const name = options.name ?? "SmartTEAM ML";
  const transport = options.transport ?? "bluetooth";
  const classesSource = generateClassesFile({ transport, classes: options.classes ?? [] });
  // Extensión del curso (si está publicada): se agrega como dependencia GitHub,
  // no inline. Cursos sin extensión publicada → solo la BLE base.
  const coursePkg: CoursePackage | undefined = options.course
    ? COURSES[options.course].package
    : undefined;

  const config = {
    name,
    dependencies: {
      core: "*",
      bluetooth: "*",
      // Extensión del curso como dependencia GitHub (clave = name del pxt.json
      // del repo, p. ej. "ext4"). El editor resuelve código + icono + locales.
      ...(coursePkg ? { [coursePkg.name]: `github:${coursePkg.repo}#${coursePkg.ref}` } : {}),
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
