// src/core/overlay/skeletonStyle.ts
//
// Estilo compartido de los esqueletos de detección (canvas no lee CSS vars de
// forma confiable a 60fps, por eso constantes TS). Paleta alineada a theme.css.

export const SKEL = {
  pink: "#FF4D8D",
  cyan: "#22D3EE",
  violet: "#A855F7",
  /** Trazo del cuerpo (pose). */
  lineWidth: 14,
  /** Trazo de manos (landmarks más densos). */
  handLineWidth: 10,
  /** Trazo de contornos de cara (malla densa: más fino). */
  faceLineWidth: 6,
  jointFill: "#FFFFFF",
  jointRadius: 8,
  handJointRadius: 6,
} as const;

export function setupStroke(ctx: CanvasRenderingContext2D, color: string, width: number) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

export function drawJoint(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  borderColor: string
) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = SKEL.jointFill;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = borderColor;
  ctx.stroke();
}
