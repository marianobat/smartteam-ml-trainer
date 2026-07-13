// src/app/components/trainer/SampleGrid.tsx
//
// Grilla de muestras de la clase activa: miniaturas (esqueleto/foto) o texto,
// tachito para borrar, y placeholders punteados hasta llegar al mínimo.

import { Trash2 } from "lucide-react";
import { COPY } from "../../copy";
import GestureIcon from "./classIcons";
import "./SampleGrid.css";

export type SampleItem = {
  id: string;
  thumb?: string;
  /** Contenido textual (modalidad textos) cuando no hay miniatura. */
  content?: string;
};

type SampleGridProps = {
  items: SampleItem[];
  min: number;
  placeholderIcon: string;
  onDelete?: (id: string) => void;
};

export default function SampleGrid({ items, min, placeholderIcon, onDelete }: SampleGridProps) {
  const placeholders = Math.max(0, min - items.length);

  return (
    <div className="sample-grid">
      {items.map((item) => (
        <div key={item.id} className="sample-card">
          {item.thumb ? (
            <img src={item.thumb} alt="" className="sample-card-img" />
          ) : item.content ? (
            <span className="sample-card-text">{item.content}</span>
          ) : (
            <span className="sample-card-icon" aria-hidden="true">
              <GestureIcon icon={placeholderIcon} size={22} />
            </span>
          )}
          {onDelete && (
            <button
              type="button"
              className="sample-card-delete"
              title={COPY.deleteSample}
              aria-label={COPY.deleteSample}
              onClick={() => onDelete(item.id)}
            >
              <Trash2 size={14} aria-hidden="true" />
            </button>
          )}
        </div>
      ))}

      {Array.from({ length: placeholders }, (_, i) => (
        <div key={`ph-${i}`} className="sample-card sample-card-placeholder" aria-hidden="true">
          {i === 0 && <span className="sample-card-next">{items.length + 1}</span>}
          <span className="sample-card-icon">
            <GestureIcon icon={placeholderIcon} size={22} />
          </span>
        </div>
      ))}
    </div>
  );
}
