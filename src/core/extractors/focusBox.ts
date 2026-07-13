// src/core/extractors/focusBox.ts
//
// Recuadro de muestreo de la modalidad IMÁGENES: ventana 4:3 centrada, nítida
// sobre el video borroso. Definición ÚNICA compartida entre la UI (CameraStage
// dibuja las franjas borrosas y el borde) y el pipeline (imageExtractor y las
// miniaturas recortan este mismo rect antes de MobileNet). Si se cambia acá,
// UI y modelo quedan alineados solos.

/** Relación de aspecto del recuadro (ancho / alto). */
export const FOCUS_BOX_ASPECT = 4 / 3;

/** Alto del recuadro relativo al alto del frame. */
export const FOCUS_BOX_HEIGHT_RATIO = 0.7;

export type FocusRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

/**
 * Rect del recuadro centrado para un frame de `width`×`height` px. Si el frame
 * es muy angosto para el 4:3 al 70% de alto, se achica manteniendo el aspecto.
 * Al estar centrado, el espejado del video no lo desplaza.
 */
export function focusBoxRect(width: number, height: number): FocusRect {
  let boxHeight = height * FOCUS_BOX_HEIGHT_RATIO;
  let boxWidth = boxHeight * FOCUS_BOX_ASPECT;
  if (boxWidth > width) {
    boxWidth = width;
    boxHeight = boxWidth / FOCUS_BOX_ASPECT;
  }
  return {
    x: (width - boxWidth) / 2,
    y: (height - boxHeight) / 2,
    width: boxWidth,
    height: boxHeight,
  };
}
