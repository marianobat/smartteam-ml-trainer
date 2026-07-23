// src/app/components/trainer/LearningCurveCard.tsx
//
// Tarjeta "Cómo va aprendiendo": la curva de precisión (entrenamiento violeta,
// validación cyan) que ocupa el escenario derecho mientras el paso 2 está
// activo. Sin textos de apoyo: solo la(s) línea(s) que crecen en vivo durante
// el entrenamiento (decisión de diseño: menos ruido para los chicos).

import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts";
import "./LearningCurveCard.css";

export type CurvePoint = {
  step: number;
  acc?: number;
  valAcc?: number;
};

type LearningCurveCardProps = {
  data: CurvePoint[];
  isTraining: boolean;
  /** Hay un modelo entrenado. */
  trainComplete: boolean;
  /** Etiqueta del eje X (se mantiene por compatibilidad; ya no se muestra). */
  xLabel?: string;
};

export default function LearningCurveCard({ data }: LearningCurveCardProps) {
  const hasData = data.length > 0;
  const hasVal = data.some((d) => d.valAcc !== undefined);

  return (
    <div className="curve-card">
      <div className="curve-card-chart">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
              <CartesianGrid stroke="var(--color-sunken)" vertical={false} />
              <XAxis dataKey="step" hide />
              <YAxis domain={[0, 1]} hide />
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
          <div className="curve-card-empty" />
        )}
      </div>
    </div>
  );
}
