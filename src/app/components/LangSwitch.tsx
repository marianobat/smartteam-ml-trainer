// src/app/components/LangSwitch.tsx
//
// Selector de idioma (ES/EN). Persiste la elección y recarga la página para
// que toda la UI (y el MakeCode embebido) tomen el idioma nuevo.

import { getLang, setLang, LANGS, type Lang } from "../i18n";
import { COPY } from "../copy";
import "./LangSwitch.css";

const LANG_LABELS: Record<Lang, string> = {
  es: "ES",
  en: "EN",
};

export default function LangSwitch() {
  const current = getLang();

  return (
    <div className="lang-switch" role="group" aria-label={COPY.langLabel}>
      {LANGS.map((lang) => (
        <button
          key={lang}
          type="button"
          className={`lang-switch-btn ${lang === current ? "is-on" : ""}`}
          aria-pressed={lang === current}
          onClick={() => setLang(lang)}
        >
          {LANG_LABELS[lang]}
        </button>
      ))}
    </div>
  );
}
