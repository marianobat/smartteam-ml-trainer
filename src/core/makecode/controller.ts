// src/core/makecode/controller.ts
//
// Hook para comunicarse con el fork de MakeCode embebido en modo controller.
// El fork (pxteditor/editorcontroller.ts) acepta requests del padre cuando
// corre con `?controller=1`. Al terminar de cargar, emite
// `{ type: "pxthost", action: "editorcontentloaded" }`; recién ahí le mandamos
// el proyecto con `{ type: "pxteditor", action: "importproject", project }`.
//
// Preservación del trabajo del alumno: `importproject` PISA el proyecto actual.
// Si en cada carga re-importáramos, el chico perdería lo que venía armando. Con
// `ImportGuard` recordamos (en localStorage) qué contenido inyectamos por
// proyecto; si es el mismo, NO re-importamos y dejamos que el workspace del
// navegador (ws=browser) reabra solo el último proyecto guardado.

import { useEffect, useRef, useState, type RefObject } from "react";
import type { MakeCodeProject } from "./project";

type EditorState = "loading" | "ready" | "imported" | "error";

interface HostMessage {
  type?: string;
  action?: string;
}

/**
 * Guarda de importación para preservar el trabajo del alumno entre cargas.
 * Si ya inyectamos el mismo `contentSig` para este `persistId` en este
 * navegador, no re-importamos (dejamos que ws=browser reabra el proyecto).
 * `null` → comportamiento clásico: importar siempre.
 */
export interface ImportGuard {
  /** Identidad estable del proyecto en este navegador (p. ej. "hands-4"). */
  persistId: string;
  /** Firma del contenido inyectable (clases entrenadas). Cambia → re-importa. */
  contentSig: string;
}

/** Prefijo de la clave localStorage donde recordamos qué contenido inyectamos. */
const GUARD_STORAGE_PREFIX = "smartteam-mk-imported-";

let messageSeq = 0;
const nextId = () => `st-${Date.now()}-${messageSeq++}`;

/**
 * Devuelve la URL del iframe en modo controller (agrega controller=1 sin pisar
 * query existente) y el origin para validar/postear mensajes.
 *
 * `ws=browser` fuerza el workspace de IndexedDB dentro del iframe. Sin esto, el
 * editor en modo controller usa el "iframe workspace", que hace un handshake de
 * storage contra el padre (workspacesync/save) y se queda colgado en el splash
 * si el padre no responde ese protocolo (nosotros sólo inyectamos importproject).
 * Además, ws=browser hace que el editor reabra solo el último proyecto guardado,
 * que es lo que aprovecha ImportGuard para no pisar el trabajo del alumno.
 */
export function resolveControllerUrl(baseUrl: string): { src: string; origin: string } | null {
  const trimmed = baseUrl.trim();
  if (!trimmed) return null;
  try {
    const url = new URL(trimmed, window.location.href);
    url.searchParams.set("controller", "1");
    url.searchParams.set("ws", "browser");
    return { src: url.toString(), origin: url.origin };
  } catch {
    return null;
  }
}

/**
 * Espera a que el editor avise que cargó y le inyecta el proyecto una sola vez.
 * `project` puede llegar async (null mientras se cargan las clases); se importa
 * cuando ambos (editor listo + proyecto) están disponibles. Si `importGuard`
 * indica que ese contenido ya se inyectó antes en este navegador, se omite la
 * importación para no pisar el proyecto que el alumno venía editando.
 */
export function useMakeCodeController(
  iframeRef: RefObject<HTMLIFrameElement | null>,
  forkOrigin: string | null,
  project: MakeCodeProject | null,
  importGuard: ImportGuard | null = null
): { state: EditorState } {
  const [state, setState] = useState<EditorState>("loading");
  const readyRef = useRef(false);
  const importedRef = useRef(false);
  const projectRef = useRef<MakeCodeProject | null>(project);
  projectRef.current = project;
  const guardRef = useRef<ImportGuard | null>(importGuard);
  guardRef.current = importGuard;

  const postAction = (action: string) => {
    const win = iframeRef.current?.contentWindow;
    if (!win || !forkOrigin) return;
    win.postMessage({ type: "pxteditor", id: nextId(), action }, forkOrigin);
  };

  // Colapsa el simulador (más espacio para los bloques). La cámara + barras del
  // trainer cumplen el rol del simulador. Se re-asegura por si el import lo expande.
  const collapseSimulator = () => {
    postAction("hidesimulator");
    window.setTimeout(() => postAction("hidesimulator"), 1200);
  };

  const sendImport = () => {
    if (importedRef.current) return;
    const win = iframeRef.current?.contentWindow;
    const proj = projectRef.current;
    if (!readyRef.current || !win || !proj || !forkOrigin) return;

    // Si ya inyectamos este mismo contenido (mismo modelo+curso+clases) en este
    // navegador, no re-importamos: dejamos que ws=browser reabra el proyecto que
    // el alumno venía armando. Si las clases cambiaron (contentSig distinto), sí
    // re-importamos con el contenido nuevo.
    const guard = guardRef.current;
    if (guard) {
      const key = GUARD_STORAGE_PREFIX + guard.persistId;
      let prev: string | null = null;
      try {
        prev = window.localStorage.getItem(key);
      } catch {
        // sin localStorage (modo privado): se cae al comportamiento de importar siempre
      }
      if (prev !== null && prev === guard.contentSig) {
        importedRef.current = true;
        setState("imported");
        collapseSimulator();
        return;
      }
      try {
        window.localStorage.setItem(key, guard.contentSig);
      } catch {
        // idem: si no se puede recordar, igual importamos abajo
      }
    }

    importedRef.current = true;
    win.postMessage(
      { type: "pxteditor", id: nextId(), action: "importproject", project: proj },
      forkOrigin
    );
    setState("imported");
    collapseSimulator();
  };

  useEffect(() => {
    if (!forkOrigin) return;
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== forkOrigin) return;
      const data = event.data as HostMessage | null;
      if (!data || data.type !== "pxthost") return;
      if (data.action === "editorcontentloaded") {
        readyRef.current = true;
        setState((s) => (s === "loading" ? "ready" : s));
        sendImport();
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forkOrigin]);

  // Si el proyecto llega después de que el editor ya estaba listo, importamos.
  useEffect(() => {
    sendImport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, forkOrigin]);

  return { state };
}
