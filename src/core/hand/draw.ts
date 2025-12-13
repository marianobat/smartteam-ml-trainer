import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

// Conexiones estándar de MediaPipe Hands (21 puntos)
const HAND_CONNECTIONS: Array<[number, number]> = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm connections
  [5, 9], [9, 13], [13, 17],
];

type Side = "Left" | "Right" | "Unknown";

function getSide(result: HandLandmarkerResult, i: number, mirrorView: boolean): Side {
  const side = (result.handedness?.[i]?.[0]?.categoryName as Side) || "Unknown";
  if (!mirrorView) return side;
  // Si la vista está espejada, invertimos para que coincida con tu percepción
  if (side === "Left") return "Right";
  if (side === "Right") return "Left";
  return "Unknown";
}

export function drawHands(
  ctx: CanvasRenderingContext2D,
  result: HandLandmarkerResult,
  opts?: { mirrorView?: boolean }
) {
  const mirrorView = opts?.mirrorView ?? true;

  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  ctx.clearRect(0, 0, w, h);

  const hands = result.landmarks ?? [];

  for (let i = 0; i < hands.length; i++) {
    const lm = hands[i];
    const side = getSide(result, i, mirrorView);

    const isLeft = side === "Left";
    const isRight = side === "Right";

    // Colores: Left = rojo, Right = azul
    // (si Unknown, gris)
    const pointColor = isLeft ? "rgba(255,0,0,0.9)"
                    : isRight ? "rgba(0,0,255,0.9)"
                    : "rgba(180,180,180,0.9)";

    const lineColor  = isLeft ? "rgba(255,0,0,0.6)"
                    : isRight ? "rgba(0,0,255,0.6)"
                    : "rgba(180,180,180,0.6)";

    // Dibujar líneas
    ctx.lineWidth = 3;
    ctx.strokeStyle = lineColor;

    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = lm[a];
      const pb = lm[b];
      ctx.beginPath();
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
      ctx.stroke();
    }

    // Dibujar puntos
    ctx.fillStyle = pointColor;
    for (const p of lm) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}