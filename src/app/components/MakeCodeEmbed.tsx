// src/app/components/MakeCodeEmbed.tsx
//
// MakeCode micro:bit embebido en un iframe (editor completo, editable). El
// alumno programa, corre el simulador y flashea por WebUSB desde dentro del
// iframe (de ahí el allow="usb"). La URL es configurable: por defecto abre el
// editor limpio; para arrancar con la extensión SmartTEAM ML Bluetooth ya
// cargada, pasá un share link (?mk=<url> o VITE_MAKECODE_URL).

// Programa SmartTEAM con la extensión ML Bluetooth ya cargada.
// Forma "#pub:<id>" → abre el editor EDITABLE in-place importando el proyecto
// (la forma "/_<id>" es la vista de solo lectura con botón Edit).
// Se puede pisar con ?mk=<url> o VITE_MAKECODE_URL.
const DEFAULT_MAKECODE_URL = "https://makecode.microbit.org/#pub:_DWv7Tw1KiTbt";

/** Resuelve la URL del editor: ?mk= (query) > VITE_MAKECODE_URL > default. */
function resolveMakeCodeUrl(): string {
  if (typeof window !== "undefined") {
    const fromQuery = new URLSearchParams(window.location.search).get("mk");
    if (fromQuery) return fromQuery;
  }
  const fromEnv = import.meta.env.VITE_MAKECODE_URL as string | undefined;
  return fromEnv && fromEnv.trim() ? fromEnv.trim() : DEFAULT_MAKECODE_URL;
}

type MakeCodeEmbedProps = {
  url?: string;
  title?: string;
};

export default function MakeCodeEmbed({ url, title = "MakeCode micro:bit" }: MakeCodeEmbedProps) {
  const src = url ?? resolveMakeCodeUrl();
  return (
    <iframe
      className="makecode-embed"
      title={title}
      src={src}
      // Permisos necesarios para flashear/usar la placa desde el iframe.
      // Sin sandbox a propósito: MakeCode (first-party, confiable) necesita
      // workers, almacenamiento, popups y descargas; el sandbox los rompe.
      allow="usb; serial; bluetooth; camera; microphone"
    />
  );
}
