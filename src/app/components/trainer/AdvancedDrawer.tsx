// src/app/components/trainer/AdvancedDrawer.tsx
//
// Cajón de "modo avanzado" (docente): los paneles técnicos viven acá.
// El contenido solo se monta cuando está abierto (recharts es caro).

import type { ReactNode } from "react";
import { COPY } from "../../copy";
import "./AdvancedDrawer.css";

type AdvancedDrawerProps = {
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
};

export default function AdvancedDrawer({ open, onToggle, children }: AdvancedDrawerProps) {
  return (
    <section className="advanced-drawer">
      <button
        type="button"
        className="advanced-drawer-toggle"
        aria-pressed={open}
        aria-expanded={open}
        onClick={onToggle}
      >
        ⚙️ {COPY.advanced} {open ? "▲" : "▼"}
      </button>
      {open && <div className="advanced-drawer-content">{children}</div>}
    </section>
  );
}
