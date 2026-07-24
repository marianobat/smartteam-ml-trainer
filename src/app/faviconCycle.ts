// src/app/faviconCycle.ts
//
// Favicon animado: rota las "S" de marca cada 3 s.

const ICONS = [
  "s-violet.png",
  "s-amber.png",
  "s-cyan.png",
  "s-lavender.png",
  "s-green.png",
  "s-coral.png",
] as const;

const INTERVAL_MS = 3000;

let started = false;

function iconUrl(name: (typeof ICONS)[number]): string {
  const base = import.meta.env.BASE_URL ?? "/";
  return `${base}brand/favicon/${name}`;
}

/** Arranca el ciclo del favicon (idempotente). */
export function startFaviconCycle(): void {
  if (typeof document === "undefined" || started) return;
  started = true;

  let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  if (!link) {
    link = document.createElement("link");
    link.rel = "icon";
    document.head.appendChild(link);
  }
  link.type = "image/png";

  for (const name of ICONS) {
    const img = new Image();
    img.src = iconUrl(name);
  }

  let index = 0;
  const apply = () => {
    link!.href = iconUrl(ICONS[index]);
  };
  apply();

  window.setInterval(() => {
    if (document.hidden) return;
    index = (index + 1) % ICONS.length;
    apply();
  }, INTERVAL_MS);
}
