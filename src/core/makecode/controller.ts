// src/core/makecode/controller.ts
//
// Hook para el MakeCode embebido en modo controller (`?controller=1`).
//
// Workspace: SIN `ws=browser`. El editor pide al padre `workspacesync` /
// `workspacesave` (ver webapp/public/controller.html de pxt). Así los bloques
// del alumno viven en localStorage del trainer y no se pierden al reimportar
// plantilla (clases / extensión).
//
// Idioma: `?lang=` según el navegador, con fallback a español.

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
 * Guarda de importación: `persistId` identifica el slot (p. ej. "hands-4");
 * `contentSig` son las clases entrenadas. Si cambia la firma, re-importamos
 * plantilla mergeada con los bloques guardados del alumno.
 */
export interface ImportGuard {
  persistId: string;
  contentSig: string;
}

/** Locales MakeCode que usamos con frecuencia en el aula. */
const MAKECODE_LANGS = new Set([
  "en",
  "es",
  "es-ES",
  "pt",
  "pt-BR",
  "fr",
  "it",
  "de",
  "ca",
  "eu",
  "gl",
]);

let messageSeq = 0;
const nextId = () => `st-${Date.now()}-${messageSeq++}`;

/**
 * Elige idioma MakeCode: preferencia del navegador si está soportada; si no, `es`.
 */
export function resolveMakeCodeLang(
  languages: readonly string[] = typeof navigator !== "undefined"
    ? navigator.languages?.length
      ? navigator.languages
      : [navigator.language]
    : ["es"]
): string {
  for (const raw of languages) {
    if (!raw) continue;
    const tag = raw.replace("_", "-");
    if (MAKECODE_LANGS.has(tag)) return tag === "es-ES" ? "es" : tag;
    const base = tag.split("-")[0]?.toLowerCase();
    if (base && MAKECODE_LANGS.has(base)) return base;
    // es-AR, es-MX, etc. → es
    if (base === "es") return "es";
    if (base === "pt") return "pt";
  }
  return "es";
}

/**
 * URL del iframe en modo controller + idioma. No usa `ws=browser`: el workspace
 * lo hostea el padre (workspacesync/save).
 */
export function resolveControllerUrl(baseUrl: string): { src: string; origin: string } | null {
  const trimmed = baseUrl.trim();
  if (!trimmed) return null;
  try {
    const url = new URL(trimmed, window.location.href);
    url.searchParams.set("controller", "1");
    url.searchParams.set("lang", resolveMakeCodeLang());
    // Quitar ws=browser si venía en la URL base: forzar workspace del padre.
    url.searchParams.delete("ws");
    return { src: url.toString(), origin: url.origin };
  } catch {
    return null;
  }
}

/**
 * Espera `editorcontentloaded`, responde workspace sync/save e inyecta el
 * proyecto plantilla (mergeado con bloques del alumno cuando corresponde).
 */
export function useMakeCodeController(
  iframeRef: RefObject<HTMLIFrameElement | null>,
  forkOrigin: string | null,
  project: MakeCodeProject | null,
  importGuard: ImportGuard | null = null
): { state: EditorState; /** true cuando el host ya escucha workspacesync (seguro cargar el iframe). */ hostReady: boolean } {
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

  const sendImport = () => {
    if (importedRef.current) return;
    const proj = projectRef.current;
    const guard = guardRef.current;
    if (!readyRef.current || !proj || !forkOrigin) return;

    const persistId = guard?.persistId;
    const contentSig = guard?.contentSig ?? "";
    const stored = persistId ? loadStudentWorkspace(persistId) : null;

    // Misma firma y ya hay proyecto en el workspace del padre → no pisar.
    // (El editor lo abrió vía workspacesync.)
    if (
      persistId &&
      stored &&
      contentSig &&
      stored.contentSig === contentSig &&
      stored.project.text
    ) {
      importedRef.current = true;
      setState("imported");
      collapseSimulator();
      return;
    }

    const text = mergeStudentFiles(proj.text, stored?.project.text);
    const toImport: MakeCodeProject = { text };

    importedRef.current = true;
    postToEditor({
      type: "pxteditor",
      id: nextId(),
      action: "importproject",
      project: toImport,
    });
    // Recordar firma + proyecto mergeado (por si workspacesave tarda).
    if (persistId) {
      saveStudentWorkspace(persistId, contentSig, { text });
    }
    setState("imported");
    collapseSimulator();
  };

  // useLayoutEffect: el listener debe existir ANTES de poner src al iframe,
  // si no se pierde el primer `workspacesync` y el editor se cuelga.
  useLayoutEffect(() => {
    if (!forkOrigin) {
      setHostReady(false);
      return;
    }

    const onMessage = (event: MessageEvent) => {
      if (event.origin !== forkOrigin) return;
      const data = event.data as HostMessage | null;
      if (!data || typeof data !== "object") return;

      // El editor pide la lista de proyectos al padre.
      if (data.type === "pxthost" && data.action === "workspacesync") {
        const persistId = guardRef.current?.persistId;
        const stored = persistId ? loadStudentWorkspace(persistId) : null;
        const projects = stored?.project ? [stored.project] : [];
        postToEditor({ ...data, projects });
        return;
      }

      // Autoguardado del editor → backup en el trainer.
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
    // Nueva firma / proyecto → permitir re-import mergeado.
    importedRef.current = false;
    sendImport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, forkOrigin, importGuard?.persistId, importGuard?.contentSig]);

  return { state, hostReady };
}
