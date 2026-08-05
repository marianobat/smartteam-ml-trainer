// src/app/i18n.ts
//
// Idioma de la plataforma. Como la app navega con recargas completas (sin
// router SPA), el diccionario se elige una vez al cargar el módulo copy.ts;
// cambiar de idioma persiste la elección y recarga la página.

export type Lang = "es" | "en" | "pt";

export const LANGS: readonly Lang[] = ["es", "en", "pt"];

/** Etiquetas cortas del desplegable (siempre en su idioma). */
export const LANG_LABELS: Record<Lang, string> = {
  es: "Español",
  en: "English",
  pt: "Português (Brasil)",
};

const STORAGE_KEY = "smartteam-lang";

export function getLang(): Lang {
  if (typeof window === "undefined") return "es";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "en" || stored === "pt" || stored === "es") return stored;
    return "es";
  } catch {
    return "es";
  }
}

/**
 * Código que entiende MakeCode (`forcelang`). BR usa `pt-BR` en pxt-microbit.
 */
export function toMakeCodeLang(lang: Lang): string {
  switch (lang) {
    case "es":
      return "es";
    case "en":
      return "en";
    case "pt":
      return "pt-BR";
    default: {
      const _exhaustive: never = lang;
      return _exhaustive;
    }
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
