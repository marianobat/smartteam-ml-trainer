// src/core/makecode/courses.ts
//
// Registro declarativo curso (3.º–9.º) → extensión MakeCode a embeber en el
// proyecto inyectado. El selector de curso de /microbit elige una entrada de
// acá y buildMakeCodeProject embebe la fuente correspondiente.
//
// CÓMO ENCHUFAR UNA EXTENSIÓN POR CURSO (cuando estén desarrolladas):
//   1. Guardar la fuente como `extensions/smartteam-ml-<curso>.ts.txt`.
//   2. Importarla abajo con `?raw` (igual que la base).
//   3. Reemplazar `extension: BLE_BASE_EXTENSION` por la nueva en COURSES.
// Mientras tanto, todos los cursos usan la extensión BLE base: el flujo queda
// idéntico al actual y las variantes se agregan sin tocar nada más.

import bleExtensionSource from "./extensions/smartteam-ml-bluetooth.ts.txt?raw";

export type CourseId = "3" | "4" | "5" | "6" | "7" | "8" | "9";

export type CourseExtension = {
  /** Nombre del archivo TS inline dentro del proyecto MakeCode. */
  file: string;
  /** Fuente completa de la extensión (importada con ?raw). */
  source: string;
};

export type Course = {
  id: CourseId;
  /** Etiqueta corta, p. ej. "3.º". */
  label: string;
  /** Etiqueta larga para la tarjeta, p. ej. "3er grado". */
  longLabel: string;
  extension: CourseExtension;
};

/** Extensión BLE actual (la misma para todos los cursos hasta tener variantes). */
const BLE_BASE_EXTENSION: CourseExtension = {
  file: "smartteamMLBT.ts",
  source: bleExtensionSource,
};

export const COURSE_IDS: readonly CourseId[] = ["3", "4", "5", "6", "7", "8", "9"];

const ORDINALS: Record<CourseId, string> = {
  "3": "3er grado",
  "4": "4to grado",
  "5": "5to grado",
  "6": "6to grado",
  "7": "7mo grado",
  "8": "8vo grado",
  "9": "9no grado",
};

export const COURSES: Record<CourseId, Course> = Object.fromEntries(
  COURSE_IDS.map((id) => [
    id,
    {
      id,
      label: `${id}.º`,
      longLabel: ORDINALS[id],
      extension: BLE_BASE_EXTENSION,
    },
  ])
) as Record<CourseId, Course>;

export function isCourseId(value: string | null | undefined): value is CourseId {
  return Boolean(value && (COURSE_IDS as readonly string[]).includes(value));
}

/** Última elección de curso (para pre-resaltar la tarjeta, no para saltear la pantalla). */
export const LAST_COURSE_STORAGE_KEY = "smartteam-microbit-last-course";
