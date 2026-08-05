// src/app/components/trainer/StepAccordion.tsx
//
// Acordeón guiado de 3 pasos (Enseñar → Entrenar → Probar) con habilitación
// paso a paso. Presentacional: el estado de cada paso (activo/completado/
// bloqueado) y cuál está abierto los deriva/decide el entrenador.
//
// Importante: los cuerpos de TODOS los pasos quedan montados (ocultos con
// `hidden` cuando el paso está colapsado) para no cortar efectos que viven
// dentro (p. ej. MicrobitPanel enviando la detección al micro:bit).

import type { ReactNode } from "react";
import { ChevronDown, ChevronUp, Lock } from "lucide-react";
import { COPY } from "../../copy";
import "./StepAccordion.css";

export type StepState = "active" | "done" | "locked";

export type AccordionStep = {
  id: string;
  title: string;
  /** Subtítulo del paso abierto (guía corta). */
  subtitle: string;
  state: StepState;
  /** Resumen mostrado cuando el paso está completado y colapsado. */
  summary?: string;
  /** Acción del paso completado ("Editar" / "Reentrenar"): reabre el paso. */
  actionLabel?: string;
  body: ReactNode;
};

type StepAccordionProps = {
  steps: AccordionStep[];
  /** Id del paso abierto (solo uno a la vez). */
  openId: string;
  onOpen: (id: string) => void;
};

export default function StepAccordion({ steps, openId, onOpen }: StepAccordionProps) {
  return (
    <ol className="step-accordion" aria-label={COPY.ariaSteps}>
      {steps.map((step, idx) => {
        const open = openId === step.id && step.state !== "locked";
        const number = idx + 1;

        if (step.state === "locked") {
          return (
            <li key={step.id} className="step-acc-item is-locked">
              <div className="step-acc-header">
                <span className="step-acc-tile" aria-hidden="true">
                  <Lock size={17} aria-hidden="true" />
                </span>
                <div className="step-acc-heading">
                  <div className="step-acc-title">
                    {number} · {step.title}
                  </div>
                </div>
              </div>
              <div hidden>{step.body}</div>
            </li>
          );
        }

        return (
          <li
            key={step.id}
            className={`step-acc-item ${open ? "is-open" : ""} ${
              step.state === "done" ? "is-done" : "is-active"
            }`}
            aria-current={step.state === "active" ? "step" : undefined}
          >
            <button
              type="button"
              className="step-acc-header step-acc-header-btn"
              aria-expanded={open}
              onClick={() => onOpen(step.id)}
            >
              <span className="step-acc-tile" aria-hidden="true">
                {number}
              </span>
              <span className="step-acc-heading">
                <span className="step-acc-title">
                  {open || step.state === "active" ? step.title : `${number} · ${step.title}`}
                </span>
                {(open ? step.subtitle : step.summary ?? step.subtitle) ? (
                  <span className="step-acc-sub">
                    {open ? step.subtitle : step.summary ?? step.subtitle}
                  </span>
                ) : null}
              </span>
              {step.state === "done" && !open && step.actionLabel ? (
                <span className="step-acc-action">{step.actionLabel}</span>
              ) : open ? (
                <ChevronUp size={20} aria-hidden="true" />
              ) : (
                <ChevronDown size={20} aria-hidden="true" />
              )}
            </button>
            <div className="step-acc-body" hidden={!open}>
              {step.body}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
