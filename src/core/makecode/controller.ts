// src/core/makecode/controller.ts
//
// Hook para comunicarse con el fork de MakeCode embebido en modo controller.
// El fork (pxteditor/editorcontroller.ts) acepta requests del padre cuando
// corre con `?controller=1`. Al terminar de cargar, emite
// `{ type: "pxthost", action: "editorcontentloaded" }`; recién ahí le mandamos
// el proyecto con `{ type: "pxteditor", action: "importproject", project }`.

import { useEffect, useRef, useState, type RefObject } from "react";
import type { MakeCodeProject } from "./project";

type EditorState = "loading" | "ready" | "imported" | "error";

interface HostMessage {
  type?: string;
  action?: string;
}

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
 * cuando ambos (editor listo + proyecto) están disponibles.
 */
export function useMakeCodeController(
  iframeRef: RefObject<HTMLIFrameElement | null>,
  forkOrigin: string | null,
  project: MakeCodeProject | null
): { state: EditorState } {
  const [state, setState] = useState<EditorState>("loading");
  const readyRef = useRef(false);
  const importedRef = useRef(false);
  const projectRef = useRef<MakeCodeProject | null>(project);
  projectRef.current = project;

  const sendImport = () => {
    if (importedRef.current) return;
    const win = iframeRef.current?.contentWindow;
    const proj = projectRef.current;
    if (!readyRef.current || !win || !proj || !forkOrigin) return;
    importedRef.current = true;
    win.postMessage(
      { type: "pxteditor", id: nextId(), action: "importproject", project: proj },
      forkOrigin
    );
    setState("imported");
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
