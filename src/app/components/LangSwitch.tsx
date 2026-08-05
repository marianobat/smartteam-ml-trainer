// src/app/components/LangSwitch.tsx
//
// Desplegable de idioma (ES / EN / PT-BR). Persiste la elección y recarga la
// página para que toda la UI (y el MakeCode embebido) tomen el idioma nuevo.

import { getLang, setLang, LANGS, LANG_LABELS, type Lang } from "../i18n";
import { COPY } from "../copy";
import "./LangSwitch.css";

export default function LangSwitch() {
  const current = getLang();

  return (
    <label className="lang-switch">
      <span className="lang-switch-sr">{COPY.langLabel}</span>
      <select
        className="lang-switch-select"
        value={current}
        aria-label={COPY.langLabel}
        onChange={(e) => setLang(e.target.value as Lang)}
      >
        {LANGS.map((lang) => (
          <option key={lang} value={lang}>
            {LANG_LABELS[lang]}
          </option>
        ))}
      </select>
    </label>
  );
}
