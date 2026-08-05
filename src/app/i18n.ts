// src/app/i18n.ts
//
// Idioma de la plataforma. Como la app navega con recargas completas (sin
// router SPA), el diccionario se elige una vez al cargar el módulo copy.ts;
// cambiar de idioma persiste la elección y recarga la página.

export type Lang = "es" | "en";

export const LANGS: readonly Lang[] = ["es", "en"];

const STORAGE_KEY = "smartteam-lang";

export function getLang(): Lang {
  if (typeof window === "undefined") return "es";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === "en" ? "en" : "es";
  } catch {
    return "es";
  }
}

/** Persiste el idioma y recarga para que toda la UI (y MakeCode) lo tomen. */
export function setLang(lang: Lang): void {
  if (lang === getLang()) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // modo privado: el cambio vale solo hasta la recarga
  }
  window.location.reload();
}
