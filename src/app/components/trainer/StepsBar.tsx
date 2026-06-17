// src/app/components/trainer/StepsBar.tsx
//
// Indicador de pasos ① Enseñale ejemplos → ② Entrená → ③ Probalo.
// Presentacional: el estado de cada paso lo deriva el entrenador.

import { Check } from "lucide-react";
import "./StepsBar.css";

export type Step = {
  label: string;
  done: boolean;
  active: boolean;
};

type StepsBarProps = {
  steps: Step[];
};

export default function StepsBar({ steps }: StepsBarProps) {
  return (
    <ol className="steps-bar" aria-label="Pasos">
      {steps.map((step, idx) => (
        <li
          key={step.label}
          className={`steps-bar-item ${step.done ? "is-done" : ""} ${step.active ? "is-active" : ""}`}
          aria-current={step.active ? "step" : undefined}
        >
          <span className="steps-bar-dot" aria-hidden="true">
            {step.done ? <Check size={16} aria-hidden="true" /> : idx + 1}
          </span>
          <span className="steps-bar-label">{step.label}</span>
          {idx < steps.length - 1 && <span className="steps-bar-line" aria-hidden="true" />}
        </li>
      ))}
    </ol>
  );
}
