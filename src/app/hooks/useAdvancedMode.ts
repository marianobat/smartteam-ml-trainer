// src/app/hooks/useAdvancedMode.ts
//
// Toggle "modo avanzado" (docente): muestra los paneles técnicos.
// Persistido en localStorage para sobrevivir recargas.

import { useCallback, useState } from "react";

const STORAGE_KEY = "st.advancedMode";

function readStored(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function useAdvancedMode(): [boolean, () => void] {
  const [advanced, setAdvanced] = useState<boolean>(readStored);

  const toggle = useCallback(() => {
    setAdvanced((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      } catch {
        // sin storage (modo incógnito estricto): el toggle vive en memoria
      }
      return next;
    });
  }, []);

  return [advanced, toggle];
}
