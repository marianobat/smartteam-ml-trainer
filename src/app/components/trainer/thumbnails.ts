// src/app/components/trainer/thumbnails.ts
//
// Miniaturas de muestras. Para modalidades con esqueleto (manos/cuerpo/cara)
// se rasteriza el canvas de overlay sobre fondo blanco — no se guarda la foto
// del chico (privacidad). Para imágenes se recorta la ventana 4:3 central
// (focusBox): la miniatura muestra exactamente lo que el modelo tomó.

import { focusBoxRect } from "../../../core/extractors/focusBox";

export function captureSkeletonThumbnail(
  overlay: HTMLCanvasElement,
  size = 96,
  mirror = true
): string {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  if (mirror) {
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
  }

  // "contain": no recorta brazos/manos que salgan del cuadrado central
  const scale = size / Math.max(overlay.width || 1, overlay.height || 1);
  const w = (overlay.width || 1) * scale;
  const h = (overlay.height || 1) * scale;
  ctx.drawImage(overlay, (size - w) / 2, (size - h) / 2, w, h);

  // PNG: trazos nítidos sobre blanco
  return c.toDataURL("image/png");
}

export function captureVideoThumbnail(
  video: HTMLVideoElement,
  size = 96,
  mirror = true
): string {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;

  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  // Cuadrado central DENTRO de la ventana 4:3 de muestreo (sin distorsión)
  const rect = focusBoxRect(vw, vh);
  const side = Math.min(rect.width, rect.height);
  const sx = rect.x + (rect.width - side) / 2;
  const sy = rect.y + (rect.height - side) / 2;

  if (mirror) {
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
  return c.toDataURL("image/jpeg", 0.7);
}
