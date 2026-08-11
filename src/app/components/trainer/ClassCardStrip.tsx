// src/app/components/trainer/ClassCardStrip.tsx
//
// Fila horizontal de tarjetas de clase (estilo LEGO): miniatura + nombre +
// progreso de ejemplos, más la tarjeta "Agregar". Presentacional.

import { Check, Plus } from "lucide-react";
import { COPY } from "../../copy";
import GestureIcon from "./classIcons";
import "./ClassCardStrip.css";

export type ClassCardItem = {
  id: string;
  name: string;
  count: number;
  thumb?: string;
  /** Ícono propio de la clase (p. ej. preset); pisa al placeholder genérico. */
  icon?: string;
};

type ClassCardStripProps = {
  items: ClassCardItem[];
  activeId: string | null;
  min: number;
  placeholderIcon: string;
  onSelect: (id: string) => void;
  onAdd: () => void;
  addDisabled?: boolean;
};

export default function ClassCardStrip({
  items,
  activeId,
  min,
  placeholderIcon,
  onSelect,
  onAdd,
  addDisabled,
}: ClassCardStripProps) {
  return (
    <div className="class-strip" role="tablist" aria-label={COPY.ariaClasses}>
      {items.map((item) => {
        const complete = item.count >= min;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={item.id === activeId}
            className={`class-strip-card ${item.id === activeId ? "is-active" : ""}`}
            onClick={() => onSelect(item.id)}
          >
            <span className="class-strip-thumb" aria-hidden="true">
              {item.thumb ? (
                <img src={item.thumb} alt="" />
              ) : item.icon ? (
                <span className="class-strip-icon">
                  <GestureIcon icon={item.icon} size={30} />
                </span>
              ) : (
                <span className="class-strip-placeholder">
                  <GestureIcon icon={placeholderIcon} size={30} />
                </span>
              )}
            </span>
            <span className="class-strip-name">{item.name || COPY.classUnnamed}</span>
            <span
              className={`class-strip-progress ${complete ? "is-complete" : ""}`}
              aria-label={COPY.examplesProgressAria(item.count, min)}
            >
              {complete ? (
                <>
                  <Check size={13} aria-hidden="true" /> {COPY.examplesCount(item.count)}
                </>
              ) : (
                `${item.count}/${min}`
              )}
            </span>
          </button>
        );
      })}

      <button
        type="button"
        className="class-strip-card class-strip-add"
        onClick={onAdd}
        disabled={addDisabled}
      >
        <span className="class-strip-thumb" aria-hidden="true">
          <span className="class-strip-placeholder">
            <Plus size={28} aria-hidden="true" />
          </span>
        </span>
        <span className="class-strip-name">{COPY.addClass}</span>
      </button>
    </div>
  );
}
