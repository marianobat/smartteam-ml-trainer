// Feature flags de producto (variables VITE_* en build time).
// Ver docs/TURBOWARP.md para reactivar la integración con TurboWarp.

const parseBool = (value: string | undefined, fallback: boolean): boolean => {
  if (value === undefined || !value.trim()) return fallback;
  const v = value.trim().toLowerCase();
  return v === "true" || v === "1" || v === "yes";
};

/** Lobby, sesión WebSocket y publicación a Scratch vía TurboWarp. Default: desactivado. */
export const TURBOWARP_ENABLED = parseBool(
  import.meta.env.VITE_ENABLE_TURBOWARP as string | undefined,
  false
);
