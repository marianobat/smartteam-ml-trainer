// src/app/components/trainer/LearningCurveCard.tsx
//
// Tarjeta "Cómo va aprendiendo": curva de precisión en el escenario del paso 2.
// Datos mínimos: leyenda de series + algunos números en la grilla.

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
  trainComplete: boolean;
  xLabel?: string;
};

export default function LearningCurveCard({ data }: LearningCurveCardProps) {
  const hasData = data.length > 0;
  const hasVal = data.some((d) => d.valAcc !== undefined);

  return (
    <div className="curve-card">
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
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 4 }}>
              <CartesianGrid stroke="var(--color-sunken)" vertical={false} />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tick={{ fill: "var(--color-ink-faint)", fontSize: 12 }}
                minTickGap={28}
              />
              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.5, 1]}
                tickLine={false}
                axisLine={false}
                width={40}
                tick={{ fill: "var(--color-ink-faint)", fontSize: 12 }}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              />
              <Line
                type="monotone"
                dataKey="acc"
                name={COPY.curveLegendTrain}
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
                  name={COPY.curveLegendVal}
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
    </div>
  );
}
