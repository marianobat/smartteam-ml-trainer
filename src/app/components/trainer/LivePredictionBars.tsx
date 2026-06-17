// src/app/components/trainer/LivePredictionBars.tsx
//
// Evaluación en vivo para chicos: "Veo: X 99%" + barras grandes por clase.

import type { CSSProperties } from "react";
import { COPY } from "../../copy";
import "./LivePredictionBars.css";

export type PredictionRow = {
  name: string;
  /** 0-1 */
  value: number;
  pass: boolean;
};

// Hue por clase (lenguaje de "bloques" de la marca): cada barra su color.
const BAR_COLORS = [
  "var(--color-secondary)", // cyan
  "var(--color-accent)", // coral/rosa
  "var(--color-primary)", // violeta de marca
  "var(--mod-images)", // naranja
  "var(--color-success)", // verde
  "var(--mod-text)", // azul
];

const barColor = (index: number) => BAR_COLORS[index % BAR_COLORS.length];

type LivePredictionBarsProps = {
  rows: PredictionRow[];
  /** Lo que el modelo "ve" ahora (null = nada confiable). */
  seeing: { label: string; confidence: number } | null;
  /** Sin modelo entrenado: mensaje vacío. */
  hasModel: boolean;
};

export default function LivePredictionBars({ rows, seeing, hasModel }: LivePredictionBarsProps) {
  if (!hasModel) {
    return <div className="live-bars-empty">{COPY.liveEmpty}</div>;
  }

  return (
    <div className="live-bars">
      <div className="live-bars-seeing" aria-live="polite">
        {seeing ? (
          <>
            {COPY.see} <strong>{seeing.label}</strong>
            <span className="live-bars-pct">{Math.round(seeing.confidence * 100)}%</span>
          </>
        ) : (
          <span className="live-bars-unsure">{COPY.seeNothing}</span>
        )}
      </div>
      <div className="live-bars-list">
        {rows.map((row, idx) => (
          <div key={row.name} className="live-bars-row">
            <span className="live-bars-name">{row.name}</span>
            <span className="live-bars-track" aria-hidden="true">
              <span
                className={`live-bars-fill ${row.pass ? "is-pass" : ""}`}
                style={
                  {
                    width: `${Math.round(Math.max(0, Math.min(1, row.value)) * 100)}%`,
                    "--bar-color": barColor(idx),
                  } as CSSProperties
                }
              />
            </span>
            <span className="live-bars-value">{Math.round(row.value * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
