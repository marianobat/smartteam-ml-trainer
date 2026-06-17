// src/app/components/pipMonitor.ts
//
// Ventana flotante de monitoreo (Document Picture-in-Picture, Chrome 116+).
// Queda always-on-top sobre MakeCode y replica el lenguaje visual de las
// páginas de entrenamiento: tema claro, la cámara con los mismos efectos
// (video atenuado + esqueleto del overlay), las barras por clase con su color,
// y los botones de conexión USB / Bluetooth del micro:bit. Comparte el contexto
// JS de la pestaña, así que lee los refs del Trainer y el store de micro:bit en
// vivo — el modelo, la cámara y la conexión siguen viviendo en la página.
//
// La PiP es un documento aparte: NO hereda las CSS variables del :root, por eso
// los colores van como hex fijos que igualan los tokens de src/theme.css.

export function isPipSupported(): boolean {
  return typeof window !== "undefined" && "documentPictureInPicture" in window;
}

export type PipPredictionRow = { name: string; value: number; pass: boolean };

export type PipMicrobitApi = {
  supported: { serial: boolean; bluetooth: boolean };
  connectUsb: () => void | Promise<void>;
  connectBle: () => void | Promise<void>;
  disconnect: () => void | Promise<void>;
  getState: () => { status: string; transport: string | null };
};

export type PipMonitorOptions = {
  /** Video de la cámara del Trainer (se espeja en un canvas). */
  video: HTMLVideoElement | null;
  /** Canvas de overlay con el esqueleto (se dibuja encima del video). */
  overlay: HTMLCanvasElement | null;
  /** Atenuar el video (modalidades con esqueleto; no en imágenes). */
  dimmed: boolean;
  title: string;
  getLabel: () => string;
  getConfidence: () => number;
  /** Hay sujeto detectado en este momento. */
  isDetecting: () => boolean;
  /** Etiqueta para mostrar cuando no hay detección, p. ej. "Sin manos". */
  missingLabel: string;
  /** Umbral para colorear la predicción como aceptada. */
  acceptThreshold: number;
  /** Filas por clase en vivo (nombre + valor 0-1 + si supera el umbral). */
  getRows: () => PipPredictionRow[];
  /** API del store de micro:bit (conectar/desconectar + estado). */
  microbit: PipMicrobitApi;
  /** Avisar al Trainer cuando la ventana se cierra (botón nativo o close()). */
  onClose: () => void;
};

// Paleta clara = tokens de src/theme.css (la PiP no lee CSS vars).
const C = {
  bg: "#f6f4ff",
  surface: "#ffffff",
  ink: "#2d2a45",
  inkSoft: "rgba(45,42,69,0.65)",
  primary: "#796eb0",
  primaryStrong: "#5f5596",
  outline: "rgba(45,42,69,0.12)",
  stage: "#14122a",
  success: "#59bb6a",
};

// Hue por clase, mismo orden que LivePredictionBars.
const BAR_COLORS = ["#35bfe9", "#ff4d8d", "#796eb0", "#ff8a3d", "#59bb6a", "#4d7cfe"];
const barColor = (i: number) => BAR_COLORS[i % BAR_COLORS.length];

// SVG inline (Lucide no se monta en el documento PiP).
const ICON_USB =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="7" r="1"/><circle cx="4" cy="20" r="1"/><path d="M4.7 19.3 19 5"/><path d="m21 3-3 1 2 2Z"/><path d="M9.26 7.68 5 12l2 5"/><path d="m10 14 5 2 3.5-3.5"/><path d="m18 12 1-1 1 1-1 1Z"/></svg>';
const ICON_BLUETOOTH =
  '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 7 10 10-5 5V2l5 5L7 17"/></svg>';

function el<K extends keyof HTMLElementTagNameMap>(
  doc: Document,
  tag: K,
  css: string,
  parent?: HTMLElement
): HTMLElementTagNameMap[K] {
  const node = doc.createElement(tag);
  node.style.cssText = css;
  if (parent) parent.appendChild(node);
  return node;
}

const PRIMARY_BTN =
  `appearance:none;border:none;cursor:pointer;border-radius:999px;padding:8px 12px;` +
  `font-weight:700;font-size:13px;background:${C.primary};color:#fff;` +
  `display:inline-flex;align-items:center;justify-content:center;gap:6px;`;
const GHOST_BTN =
  `appearance:none;cursor:pointer;border-radius:999px;padding:8px 12px;font-weight:700;` +
  `font-size:13px;background:${C.surface};color:${C.ink};border:1.5px solid ${C.outline};` +
  `display:inline-flex;align-items:center;justify-content:center;gap:6px;`;

/** Abre la ventana de monitoreo y devuelve una función para cerrarla. */
export async function openPipMonitor(opts: PipMonitorOptions): Promise<() => void> {
  const api = window.documentPictureInPicture;
  if (!api) {
    throw new Error("Este navegador no soporta la ventana de monitoreo (Chrome 116+).");
  }

  const pip = await api.requestWindow({ width: 360, height: 540 });
  const doc = pip.document;
  doc.title = `Monitoreo — ${opts.title}`;
  doc.body.style.cssText =
    `margin:0;padding:12px;background:${C.bg};color:${C.ink};box-sizing:border-box;` +
    `font-family:"Nunito",system-ui,sans-serif;display:grid;gap:10px;align-content:start;`;

  // Título
  const titleEl = el(doc, "div", `font-size:12px;color:${C.inkSoft};font-weight:700;`, doc.body);
  titleEl.textContent = opts.title;

  // Escenario de cámara (mismos efectos que CameraStage: redondeado, video
  // atenuado + esqueleto encima).
  const stage = el(
    doc,
    "div",
    `position:relative;border-radius:18px;overflow:hidden;background:${C.stage};` +
      `box-shadow:0 6px 20px rgba(70,50,140,0.12);line-height:0;`,
    doc.body
  );
  const canvas = el(
    doc,
    "canvas",
    "width:100%;display:block;transform:scaleX(-1);",
    stage
  ) as HTMLCanvasElement;
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext("2d");

  // Headline "Veo: X 99%"
  const seeingEl = el(
    doc,
    "div",
    `font-family:"Fredoka","Nunito",system-ui,sans-serif;font-size:20px;font-weight:700;` +
      `min-height:26px;display:flex;align-items:baseline;gap:6px;`,
    doc.body
  );

  // Barras por clase
  const barsEl = el(doc, "div", "display:grid;gap:6px;", doc.body);
  const emptyEl = el(
    doc,
    "div",
    `font-size:13px;color:${C.inkSoft};background:#ebe9f6;border-radius:14px;` +
      `padding:10px;text-align:center;`,
    doc.body
  );
  emptyEl.textContent = "Cuando entrenes tu modelo, aquí vas a ver qué detecta en vivo.";

  type BarRow = { name: HTMLElement; fill: HTMLElement; value: HTMLElement; root: HTMLElement };
  let barRows: BarRow[] = [];
  let barNames: string[] = [];

  function rebuildBars(rows: PipPredictionRow[]) {
    barsEl.replaceChildren();
    barRows = rows.map((row, i) => {
      const root = el(
        doc,
        "div",
        "display:grid;grid-template-columns:minmax(56px,90px) 1fr 40px;align-items:center;gap:8px;",
        barsEl
      );
      const name = el(
        doc,
        "span",
        "font-weight:700;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
        root
      );
      name.textContent = row.name;
      const track = el(
        doc,
        "span",
        `display:block;height:16px;border-radius:999px;background:${C.outline};overflow:hidden;`,
        root
      );
      const fill = el(
        doc,
        "span",
        `display:block;height:100%;border-radius:999px;background:${barColor(i)};` +
          `width:0%;transition:width 120ms ease,opacity 120ms ease;`,
        track
      );
      const value = el(
        doc,
        "span",
        `font-family:"Fredoka","Nunito",system-ui,sans-serif;font-size:13px;text-align:right;` +
          "font-variant-numeric:tabular-nums;",
        root
      );
      return { name, fill, value, root };
    });
    barNames = rows.map((r) => r.name);
  }

  // Sección micro:bit
  const mbWrap = el(
    doc,
    "div",
    `display:grid;gap:8px;border-top:1.5px solid ${C.outline};padding-top:10px;`,
    doc.body
  );
  const mbConnectRow = el(doc, "div", "display:flex;gap:8px;", mbWrap);
  let usbBtn: HTMLButtonElement | null = null;
  let bleBtn: HTMLButtonElement | null = null;
  if (opts.microbit.supported.serial) {
    usbBtn = el(doc, "button", `${PRIMARY_BTN}flex:1;`, mbConnectRow) as HTMLButtonElement;
    usbBtn.innerHTML = `${ICON_USB}<span>USB</span>`;
    usbBtn.addEventListener("click", () => void opts.microbit.connectUsb());
  }
  if (opts.microbit.supported.bluetooth) {
    bleBtn = el(doc, "button", `${PRIMARY_BTN}flex:1;`, mbConnectRow) as HTMLButtonElement;
    bleBtn.innerHTML = `${ICON_BLUETOOTH}<span>Bluetooth</span>`;
    bleBtn.addEventListener("click", () => void opts.microbit.connectBle());
  }
  const mbDisconnectBtn = el(doc, "button", `${GHOST_BTN}width:100%;`, mbWrap) as HTMLButtonElement;
  mbDisconnectBtn.textContent = "Desconectar micro:bit";
  mbDisconnectBtn.addEventListener("click", () => void opts.microbit.disconnect());
  const mbStatus = el(doc, "div", `font-size:12px;color:${C.inkSoft};`, mbWrap);

  if (!opts.microbit.supported.serial && !opts.microbit.supported.bluetooth) {
    mbWrap.style.display = "none";
  }

  let raf = 0;
  let closed = false;

  const tick = () => {
    if (closed) return;

    // Imagen: mismos efectos que CameraStage (video atenuado + esqueleto).
    const video = opts.video;
    if (ctx && video && video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.filter = opts.dimmed ? "saturate(0.35) brightness(0.8) contrast(0.95)" : "none";
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.filter = "none";
      const overlay = opts.overlay;
      if (overlay && overlay.width > 0) {
        ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
      }
    }

    const rows = opts.getRows();
    const detecting = opts.isDetecting();
    const label = detecting ? opts.getLabel() || opts.missingLabel : opts.missingLabel;
    const confidence = detecting ? opts.getConfidence() : 0;
    const seeing =
      rows.length > 0 &&
      detecting &&
      confidence >= opts.acceptThreshold &&
      label &&
      label !== opts.missingLabel;

    // Headline "Veo: X 99%"
    if (seeing) {
      seeingEl.innerHTML =
        `<span>Veo:</span><strong style="color:${C.primary}">${escapeHtml(label)}</strong>` +
        `<span style="font-size:15px;color:${C.inkSoft}">${Math.round(confidence * 100)}%</span>`;
    } else {
      seeingEl.innerHTML = `<span style="font-size:15px;color:${C.inkSoft}">No estoy seguro todavía...</span>`;
    }

    // Barras por clase
    emptyEl.style.display = rows.length ? "none" : "block";
    barsEl.style.display = rows.length ? "grid" : "none";
    const sameRows = rows.length === barNames.length && rows.every((r, i) => r.name === barNames[i]);
    if (!sameRows) rebuildBars(rows);
    rows.forEach((row, i) => {
      const br = barRows[i];
      if (!br) return;
      const pct = Math.round(Math.max(0, Math.min(1, row.value)) * 100);
      br.fill.style.width = `${pct}%`;
      br.fill.style.opacity = row.pass ? "1" : "0.55";
      br.fill.style.boxShadow = row.pass ? `0 0 0 2px ${barColor(i)}55` : "none";
      br.value.textContent = `${pct}%`;
    });

    // micro:bit
    const st = opts.microbit.getState();
    const open = st.status === "open";
    const busy = st.status === "disconnecting";
    const connecting = st.status === "connecting";
    mbConnectRow.style.display = open || busy ? "none" : "flex";
    mbDisconnectBtn.style.display = open || busy ? "block" : "none";
    mbDisconnectBtn.disabled = busy;
    mbDisconnectBtn.textContent = busy ? "Desconectando..." : "Desconectar micro:bit";
    if (usbBtn) usbBtn.disabled = connecting;
    if (bleBtn) bleBtn.disabled = connecting;
    const transportLabel =
      st.transport === "bluetooth" ? "Bluetooth" : st.transport === "serial" ? "USB" : "";
    mbStatus.innerHTML = open
      ? `<b style="color:${C.success}">●</b> micro:bit conectado (${transportLabel})`
      : connecting
      ? "Conectando..."
      : st.status === "error"
      ? `<b style="color:#ff5a5a">●</b> error de conexión`
      : "○ micro:bit desconectado";

    raf = pip.requestAnimationFrame(tick);
  };
  raf = pip.requestAnimationFrame(tick);

  const cleanup = () => {
    if (closed) return;
    closed = true;
    pip.cancelAnimationFrame(raf);
    opts.onClose();
  };

  pip.addEventListener("pagehide", cleanup);

  return () => {
    cleanup();
    try {
      pip.close();
    } catch {
      // la ventana ya estaba cerrada
    }
  };
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === '"' ? "&quot;" : "&#39;"
  );
}
