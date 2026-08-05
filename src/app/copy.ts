// src/app/copy.ts
//
// Diccionario activo de la plataforma. El idioma se decide UNA vez al cargar
// el módulo (getLang, persistido en localStorage); cambiarlo recarga la página
// (ver i18n.ts), así todos los consumidores de COPY quedan consistentes.
//
// Los textos viven en copy.es.ts (base) y copy.en.ts; `AppCopy` obliga a que
// todos los idiomas tengan exactamente las mismas claves.

import { getLang } from "./i18n";
import { COPY_ES, type AppCopy } from "./copy.es";
import { COPY_EN } from "./copy.en";

export type { AppCopy };

const DICTIONARIES: Record<ReturnType<typeof getLang>, AppCopy> = {
  es: COPY_ES,
  en: COPY_EN,
};

export const COPY: AppCopy = DICTIONARIES[getLang()];
