// src/app/components/ProjectPanel.tsx
//
// Panel "Proyecto" compartido por los entrenadores: muestra el estado del
// auto-guardado y ofrece exportar/importar ZIP y borrar el proyecto guardado.

import { useRef } from "react";
import { Check } from "lucide-react";
import { COPY } from "../copy";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

type ProjectPanelProps = {
  saveStatus: SaveStatus;
  savedAt: number | null;
  /** Hay algo que valga la pena exportar (clases con muestras o modelo). */
  canExport: boolean;
  error?: string | null;
  onExport: () => void;
  onImport: (file: File) => void;
  onClear: () => void;
};

function formatTime(ts: number): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

export default function ProjectPanel({
  saveStatus,
  savedAt,
  canExport,
  error,
  onExport,
  onImport,
  onClear,
}: ProjectPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const statusLabel =
    saveStatus === "saving" ? (
      COPY.chipSaving
    ) : saveStatus === "saved" ? (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
        <Check size={13} aria-hidden="true" /> {COPY.chipSaved}
        {savedAt ? ` ${formatTime(savedAt)}` : ""}
      </span>
    ) : saveStatus === "error" ? (
      COPY.projSaveFailed
    ) : (
      COPY.projUnsaved
    );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onImport(file);
    event.target.value = ""; // permitir re-importar el mismo archivo
  };

  const handleClear = () => {
    const ok = window.confirm(COPY.projClearConfirm);
    if (ok) onClear();
  };

  return (
    <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>{COPY.projTitle}</div>
        <div style={{ fontSize: 12, opacity: 0.8 }}>{statusLabel}</div>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={onExport} disabled={!canExport} style={{ flex: 1 }}>
          {COPY.projExport}
        </button>
        <button onClick={() => fileInputRef.current?.click()} style={{ flex: 1 }}>
          {COPY.projImport}
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,application/zip"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      <button onClick={handleClear} style={{ fontSize: 12 }}>
        {COPY.projClear}
      </button>
      {error && <div style={{ fontSize: 12, color: "#b91c1c" }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.65 }}>{COPY.projNote}</div>
    </div>
  );
}
