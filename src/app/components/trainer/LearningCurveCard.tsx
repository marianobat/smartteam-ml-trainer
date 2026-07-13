// src/app/components/trainer/LearningCurveCard.tsx
//
// Tarjeta "Cómo va aprendiendo": la curva de precisión (entrenamiento violeta,
// validación cyan) que ocupa el escenario derecho mientras el paso 2 está
// activo. Reutiliza el mismo lineData/trainHistory del cajón avanzado, pero
// con lenguaje para chicos. Crece en vivo durante el entrenamiento.

import { TrendingUp, Info } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts";
import { COPY } from "../../copy";
import "./LearningCurveCard.css";

export type CurvePoint = {
  step: number;
  acc?: number;
  valAcc?: number;
};

type LearningCurveCardProps = {
  data: CurvePoint[];
  isTraining: boolean;
  /** Hay un modelo entrenado (muestra el chip "Listo" al terminar). */
  trainComplete: boolean;
  /** Etiqueta del eje X ("Cantidad de ejemplos…" / épocas). */
  xLabel?: string;
};

export default function LearningCurveCard({
  data,
  isTraining,
  trainComplete,
  xLabel = COPY.curveXLabel,
}: LearningCurveCardProps) {
  const hasData = data.length > 0;
  const hasVal = data.some((d) => d.valAcc !== undefined);

  return (
    <div className="curve-card">
      <div className="curve-card-header">
        <span className="curve-card-tile" aria-hidden="true">
          <TrendingUp size={24} aria-hidden="true" />
        </span>
        <div className="curve-card-heading">
          <div className="curve-card-title">{COPY.curveTitle}</div>
          <div className="curve-card-subtitle">{COPY.curveSubtitle}</div>
        </div>
        {(isTraining || trainComplete) && (
          <span className={`curve-card-chip ${isTraining ? "is-live" : ""}`}>
            <span className="curve-card-dot" aria-hidden="true" />
            {isTraining ? COPY.curveTraining : COPY.curveDone}
          </span>
        )}
      </div>

      <div className="curve-card-legend">
        <span className="curve-card-legend-item">
          <span className="curve-card-swatch" style={{ background: "var(--color-primary)" }} />
          {COPY.curveLegendTrain}
        </span>
        {hasVal && (
          <span className="curve-card-legend-item">
            <span className="curve-card-swatch" style={{ background: "var(--color-secondary)" }} />
            {COPY.curveLegendVal}
          </span>
        )}
      </div>

      <div className="curve-card-chart">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="var(--color-sunken)" vertical={false} />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tick={{ fill: "var(--color-ink-faint)", fontSize: 13 }}
                label={{
                  value: xLabel,
                  position: "insideBottom",
                  offset: -2,
                  fill: "var(--color-ink-faint)",
                  fontSize: 13,
                }}
                height={40}
              />
              <YAxis
                domain={[0, 1]}
                tickCount={4}
                tickLine={false}
                axisLine={false}
                tick={{ fill: "var(--color-ink-faint)", fontSize: 13 }}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              />
              <Line
                type="monotone"
                dataKey="acc"
                stroke="var(--color-primary)"
                strokeWidth={4.5}
                strokeLinecap="round"
                dot={false}
                isAnimationActive={false}
              />
              {hasVal && (
                <Line
                  type="monotone"
                  dataKey="valAcc"
                  stroke="var(--color-secondary)"
                  strokeWidth={4.5}
                  strokeLinecap="round"
                  dot={false}
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="curve-card-empty">{COPY.curveEmpty}</div>
        )}
      </div>

      <div className="curve-card-note">
        <span className="curve-card-note-tile" aria-hidden="true">
          <Info size={18} aria-hidden="true" />
        </span>
        {COPY.curveNote}
      </div>
    </div>
  );
}
