// src/core/makecode/studentWorkspace.ts
//
// Backup del programa del alumno (main.blocks / main.ts) en localStorage del
// shell. La plantilla (pxt.json con extensiones, BLE, clases.ts) SIEMPRE sale
// de buildMakeCodeProject; acá solo se preservan los archivos que el chico edita.

export type MakeCodeWorkspaceProject = {
  text?: Record<string, string>;
  header?: Record<string, unknown>;
  [key: string]: unknown;
};

export type StoredStudentWorkspace = {
  contentSig: string;
  project: MakeCodeWorkspaceProject;
  savedAt: number;
};

const STORAGE_PREFIX = "smartteam-mk-ws-";

function storageKey(persistId: string): string {
  return STORAGE_PREFIX + persistId;
}

export function loadStudentWorkspace(persistId: string): StoredStudentWorkspace | null {
  try {
    const raw = window.localStorage.getItem(storageKey(persistId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredStudentWorkspace;
    if (!parsed?.project?.text) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function saveStudentWorkspace(
  persistId: string,
  contentSig: string,
  project: MakeCodeWorkspaceProject
): void {
  if (!project?.text) return;
  try {
    const payload: StoredStudentWorkspace = {
      contentSig,
      project,
      savedAt: Date.now(),
    };
    window.localStorage.setItem(storageKey(persistId), JSON.stringify(payload));
  } catch {
    // cuota / modo privado: no rompemos el editor
  }
}

/** XML sin ningún `<block>`: el alumno no tiene programa propio guardado. */
function hasBlocks(blocksXml: string | undefined): blocksXml is string {
  return typeof blocksXml === "string" && blocksXml.includes("<block");
}

/**
 * Plantilla actual (deps/extensiones/clases) + bloques del alumno.
 * Nunca reutiliza el pxt.json viejo del backup: las extensiones vienen siempre
 * de la plantilla fresca. Un `main.blocks` sin bloques no cuenta como programa:
 * se prefiere el canvas starter de la plantilla ("al iniciar" + "para siempre").
 */
export function mergeStudentFiles(
  templateText: Record<string, string>,
  studentText: Record<string, string> | undefined
): Record<string, string> {
  if (!studentText) return { ...templateText };
  const studentBlocks = studentText["main.blocks"];
  // Sin bloques propios → plantilla entera (main.ts del alumno quedaría
  // desincronizado con el canvas starter).
  if (!hasBlocks(studentBlocks)) return { ...templateText };
  return {
    ...templateText,
    "main.blocks": studentBlocks,
    "main.ts": studentText["main.ts"] ?? templateText["main.ts"],
  };
}
