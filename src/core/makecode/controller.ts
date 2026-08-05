// src/core/makecode/controller.ts
//
// MakeCode embebido en modo controller (`?controller=1`).
//
// - Idioma: `lang` + `forcelang` = idioma elegido en el trainer (app/i18n.ts).
// - Workspace: SIN `ws=browser`. El padre responde `workspacesync` /
//   `workspacesave` (como webapp/public/controller.html de pxt) para persistir
//   los bloques del alumno en localStorage del trainer.
// - Extensiones: SIEMPRE se hace `importproject` con plantilla fresca
//   (pxt.json + deps GitHub + BLE + clases) mergeada con main.blocks/main.ts
//   del alumno. Nunca se salta el import (saltar dejaba un proyecto viejo sin
//   las extensiones de curso).

import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import type { MakeCodeProject } from "./project";
import {
  loadStudentWorkspace,
  mergeStudentFiles,
  saveStudentWorkspace,
  type MakeCodeWorkspaceProject,
} from "./studentWorkspace";

type EditorState = "loading" | "ready" | "imported" | "error";

interface HostMessage {
  type?: string;
  action?: string;
  id?: string;
  projects?: MakeCodeWorkspaceProject[];
  project?: MakeCodeWorkspaceProject;
  editor?: unknown;
}

/**
 * Slot de persistencia del alumno (p. ej. "hands-4") + firma de clases.
 * Si cambia contentSig, se re-importa plantilla mergeada con los bloques guardados.
 */
export interface ImportGuard {
  persistId: string;
  contentSig: string;
}

let messageSeq = 0;
const nextId = () => `st-${Date.now()}-${messageSeq++}`;

/**
 * URL del iframe: controller + idioma del trainer. Sin ws=browser (workspace
 * lo hostea el padre).
 */
export function resolveControllerUrl(
  baseUrl: string,
  lang: string = "es"
): { src: string; origin: string } | null {
  const trimmed = baseUrl.trim();
  if (!trimmed) return null;
  try {
    const url = new URL(trimmed, window.location.href);
    url.searchParams.set("controller", "1");
    // forcelang pisa la preferencia guardada del editor (cookie PXT_LANG);
    // lang solo aplica si el usuario no tiene idioma elegido.
    url.searchParams.set("lang", lang);
    url.searchParams.set("forcelang", lang);
    url.searchParams.delete("ws");
    return { src: url.toString(), origin: url.origin };
  } catch {
    return null;
  }
}

/**
 * Escucha workspacesync/save, marca hostReady (para poner src al iframe) e
 * inyecta siempre el proyecto mergeado cuando el editor y la plantilla están listos.
 */
export function useMakeCodeController(
  iframeRef: RefObject<HTMLIFrameElement | null>,
  forkOrigin: string | null,
  project: MakeCodeProject | null,
  importGuard: ImportGuard | null = null
): { state: EditorState; hostReady: boolean } {
  const [state, setState] = useState<EditorState>("loading");
  const [hostReady, setHostReady] = useState(false);
  const readyRef = useRef(false);
  const importedRef = useRef(false);
  const projectRef = useRef<MakeCodeProject | null>(project);
  projectRef.current = project;
  const guardRef = useRef<ImportGuard | null>(importGuard);
  guardRef.current = importGuard;

  const postToEditor = (msg: object) => {
    const win = iframeRef.current?.contentWindow;
    if (!win || !forkOrigin) return;
    win.postMessage(msg, forkOrigin);
  };

  const postAction = (action: string) => {
    postToEditor({ type: "pxteditor", id: nextId(), action });
  };

  const collapseSimulator = () => {
    postAction("hidesimulator");
    window.setTimeout(() => postAction("hidesimulator"), 1200);
  };

  /** Plantilla actual + bloques del alumno (si hay backup). */
  const buildMergedProject = (): MakeCodeProject | null => {
    const proj = projectRef.current;
    if (!proj) return null;
    const persistId = guardRef.current?.persistId;
    const stored = persistId ? loadStudentWorkspace(persistId) : null;
    return { text: mergeStudentFiles(proj.text, stored?.project.text) };
  };

  const sendImport = () => {
    if (importedRef.current) return;
    if (!readyRef.current || !forkOrigin) return;
    const toImport = buildMergedProject();
    if (!toImport) return;

    importedRef.current = true;
    postToEditor({
      type: "pxteditor",
      id: nextId(),
      action: "importproject",
      project: toImport,
    });

    // El backup se escribe SOLO desde workspacesave (lo que el alumno edita).
    // Guardar acá pisaba el slot con la plantilla (canvas starter) en la
    // primera visita, y ese backup "vacío" ganaba en merges posteriores.
    setState("imported");
    collapseSimulator();
  };

  // Listener ANTES de asignar src al iframe (si no, se pierde el primer workspacesync).
  useLayoutEffect(() => {
    if (!forkOrigin) {
      setHostReady(false);
      return;
    }

    const onMessage = (event: MessageEvent) => {
      if (event.origin !== forkOrigin) return;
      const data = event.data as HostMessage | null;
      if (!data || typeof data !== "object") return;

      if (data.type === "pxthost" && data.action === "workspacesync") {
        // Responder con proyecto mergeado (plantilla fresca + bloques) o vacío.
        const merged = buildMergedProject();
        const projects = merged ? [{ text: merged.text }] : [];
        postToEditor({ ...data, projects });
        return;
      }

      if (data.type === "pxthost" && data.action === "workspacesave" && data.project) {
        const persistId = guardRef.current?.persistId;
        if (persistId) {
          saveStudentWorkspace(
            persistId,
            guardRef.current?.contentSig ?? "",
            data.project
          );
        }
        return;
      }

      if (data.type === "pxthost" && data.action === "editorcontentloaded") {
        readyRef.current = true;
        setState((s) => (s === "loading" ? "ready" : s));
        // Siempre importar: garantiza pxt.json con extensiones de curso.
        sendImport();
      }
    };

    window.addEventListener("message", onMessage);
    setHostReady(true);
    return () => {
      window.removeEventListener("message", onMessage);
      setHostReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forkOrigin]);

  useEffect(() => {
    // Nueva plantilla / firma → volver a importar (merge con bloques guardados).
    importedRef.current = false;
    sendImport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, forkOrigin, importGuard?.persistId, importGuard?.contentSig]);

  return { state, hostReady };
}
