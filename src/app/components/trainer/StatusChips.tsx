// src/app/components/trainer/StatusChips.tsx
//
// Chips de estado compactos (guardado ✓ / TurboWarp / micro:bit).

import type { ReactNode } from "react";
import "./StatusChips.css";

export type StatusChip = {
  id: string;
  icon: ReactNode;
  label: string;
  tone: "ok" | "off" | "warn";
};

type StatusChipsProps = {
  chips: StatusChip[];
};

export default function StatusChips({ chips }: StatusChipsProps) {
  return (
    <div className="status-chips">
      {chips.map((chip) => (
        <span key={chip.id} className={`status-chip tone-${chip.tone}`}>
          <span className="status-chip-icon" aria-hidden="true">
            {chip.icon}
          </span>
          {chip.label}
        </span>
      ))}
    </div>
  );
}
