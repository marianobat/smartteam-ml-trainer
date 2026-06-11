// src/app/components/ProjectPanel.tsx
//
// Panel "Proyecto" compartido por los entrenadores: muestra el estado del
// auto-guardado y ofrece exportar/importar ZIP y borrar el proyecto guardado.

import { useRef } from "react";

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
    saveStatus === "saving"
      ? "Guardando..."
      : saveStatus === "saved"
      ? `Guardado ✓${savedAt ? ` ${formatTime(savedAt)}` : ""}`
      : saveStatus === "error"
      ? "Error al guardar"
      : "Sin guardar";

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onImport(file);
    event.target.value = ""; // permitir re-importar el mismo archivo
  };

  const handleClear = () => {
    const ok = window.confirm(
      "¿Borrar el proyecto guardado? Se pierden las clases, muestras y el modelo entrenado de esta modalidad."
    );
    if (ok) onClear();
  };

  return (
    <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 12, fontWeight: 600 }}>Proyecto</div>
        <div style={{ fontSize: 12, opacity: 0.8 }}>{statusLabel}</div>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={onExport} disabled={!canExport} style={{ flex: 1 }}>
          Exportar ZIP
        </button>
        <button onClick={() => fileInputRef.current?.click()} style={{ flex: 1 }}>
          Importar ZIP
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
        Borrar proyecto guardado
      </button>
      {error && <div style={{ fontSize: 12, color: "#b91c1c" }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.65 }}>
        El proyecto se guarda solo en este navegador. Usá el ZIP para llevarlo a otra computadora.
      </div>
    </div>
  );
}
