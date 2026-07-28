// src/core/makecode/courses.ts
//
// Registro declarativo curso (3.º–9.º) → extensión MakeCode a agregar al
// proyecto inyectado. El selector de curso de /microbit elige una entrada de
// acá y buildMakeCodeProject agrega la extensión como DEPENDENCIA de GitHub en
// el pxt.json (no inline): así soporta extensiones multi-archivo, con icono y
// locales, sin reimplementar la resolución de paquetes de pxt.
//
// Requiere que el editor embebido resuelva paquetes de GitHub (`/api/gh`), que
// es el caso del editor oficial (default de MAKECODE_FORK_URL). La extensión
// BLE base (smartteamMLBT) sigue yendo inline aparte (ver project.ts).
//
// CÓMO ENCHUFAR LA EXTENSIÓN DE UN CURSO:
//   1. Publicar el repo en GitHub (público) y crear un RELEASE/tag (p. ej. v2.0.10).
//   2. Poner acá el paquete: { name, repo, ref }.
//      - `name` = campo "name" del pxt.json de la extensión (clave de dependencia).
//      - `repo` = "owner/repo".
//      - `ref`  = tag/release a fijar (reproducible en el aula; evitar branch).
// Sin paquete (undefined), el curso queda con solo la extensión BLE base
// (comportamiento actual), sin romper el flujo.

export type CourseId = "3" | "4" | "5" | "6" | "7" | "8" | "9";

/** Extensión de curso publicada en GitHub, agregada como dependencia del proyecto. */
export type CoursePackage = {
  /** Clave de dependencia en pxt.json (= campo "name" del pxt.json de la extensión). */
  name: string;
  /** Referencia GitHub "owner/repo". */
  repo: string;
  /** Tag/release a fijar (idealmente un release; reproducible). */
  ref: string;
};

export type Course = {
  id: CourseId;
  /** Etiqueta corta, p. ej. "3º". */
  label: string;
  /** Etiqueta larga para la tarjeta, p. ej. "3er grado". */
  longLabel: string;
  /** Extensión del curso; undefined mientras no esté publicada (solo BLE base). */
  package?: CoursePackage;
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

/**
 * Paquete GitHub por curso. undefined = todavía no publicado (usa solo la BLE
 * base). Al publicar cada extensión, completar acá con { name, repo, ref }.
 *
 * ext4: publicado como release v2.0.10 (github.com/smartteamok/smartteam-ml-ext4).
 * OJO: el `name` de dependencia es "ext4" (campo name del pxt.json del repo),
 * no el nombre del repo. Hay trabajo local sin commitear que sube a 2.0.11;
 * cuando se publique ese release, bumpear `ref` acá.
 */
const COURSE_PACKAGES: Partial<Record<CourseId, CoursePackage>> = {
  "4": { name: "ext4", repo: "LOGOS-SmartTEAM/EXT4", ref: "main" },
  "5": { name: "ext5", repo: "LOGOS-SmartTEAM/EXT5", ref: "main" },
  "6": { name: "ext6", repo: "LOGOS-SmartTEAM/EXT6", ref: "main" },
  "7": { name: "ext7", repo: "LOGOS-SmartTEAM/EXT7", ref: "main" },
  "8": { name: "ext8", repo: "LOGOS-SmartTEAM/EXT8", ref: "main" },
  "9": { name: "ext9", repo: "LOGOS-SmartTEAM/EXT9", ref: "main" },
};

export const COURSES: Record<CourseId, Course> = Object.fromEntries(
  COURSE_IDS.map((id) => [
    id,
    {
      id,
      label: `${id}º`,
      longLabel: ORDINALS[id],
      package: COURSE_PACKAGES[id],
    },
  ])
) as Record<CourseId, Course>;

export function isCourseId(value: string | null | undefined): value is CourseId {
  return Boolean(value && (COURSE_IDS as readonly string[]).includes(value));
}

/** Última elección de curso (para pre-resaltar la tarjeta, no para saltear la pantalla). */
export const LAST_COURSE_STORAGE_KEY = "smartteam-microbit-last-course";
