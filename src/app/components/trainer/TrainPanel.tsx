// src/app/components/trainer/TrainPanel.tsx
//
// Botón grande de entrenar con barra de progreso y mensajes para chicos.

import { Sparkles } from "lucide-react";
import { COPY } from "../../copy";
import "./TrainPanel.css";

type TrainPanelProps = {
  canTrain: boolean;
  isTraining: boolean;
  trainComplete: boolean;
  /** 0-100 mientras entrena (null = indeterminado). */
  progressPct: number | null;
  /** Por qué no se puede entrenar todavía (se muestra bajo el botón). */
  hint: string | null;
  error?: string | null;
  onTrain: () => void;
};

export default function TrainPanel({
  canTrain,
  isTraining,
  progressPct,
  hint,
  error,
  onTrain,
}: TrainPanelProps) {
  return (
    <div className="train-panel">
      <button
        type="button"
        className="train-panel-btn"
        disabled={!canTrain || isTraining}
        onClick={onTrain}
      >
        {isTraining ? (
          COPY.training
        ) : (
          <>
            <Sparkles size={18} aria-hidden="true" />
            {COPY.train}
          </>
        )}
      </button>
      {isTraining && (
        <div className="train-panel-progress" role="progressbar" aria-label={COPY.ariaTraining}>
          <div
            className={`train-panel-progress-fill ${progressPct === null ? "is-indeterminate" : ""}`}
            style={progressPct !== null ? { width: `${progressPct}%` } : undefined}
          />
        </div>
      )}
      {!isTraining && hint && <div className="train-panel-hint">{hint}</div>}
      {error && <div className="train-panel-error">{error}</div>}
    </div>
  );
}
