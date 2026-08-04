// src/core/makecode/studentWorkspace.ts
//
// Persistencia del proyecto MakeCode del alumno en localStorage del shell.
// El editor en modo controller (sin ws=browser) pide workspacesync / manda
// workspacesave; nosotros guardamos el proyecto completo (incl. main.blocks).

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
    // cuota / modo privado: se pierde el backup, no rompemos el editor
  }
}

/**
 * Combina plantilla (extensiones + clases) con los bloques que el alumno ya tenía.
 */
export function mergeStudentFiles(
  templateText: Record<string, string>,
  studentText: Record<string, string> | undefined
): Record<string, string> {
  if (!studentText) return { ...templateText };
  return {
    ...templateText,
    "main.blocks": studentText["main.blocks"] ?? templateText["main.blocks"],
    "main.ts": studentText["main.ts"] ?? templateText["main.ts"],
  };
}
