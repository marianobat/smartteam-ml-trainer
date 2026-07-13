// src/app/components/trainer/classIcons.tsx
//
// Íconos de línea (Lucide + SVG propios del sistema SmartTEAM) para clases y
// placeholders, reemplazando los emojis en la UI. Los datos (presets, configs)
// siguen usando el emoji como CLAVE — acá solo se resuelve su representación
// visual, así no se toca el contrato de nombres canónicos ni la persistencia.

import type { CSSProperties, ReactElement } from "react";
import {
  Hand,
  ThumbsUp,
  ThumbsDown,
  Pointer,
  PersonStanding,
  Image,
  Smile,
  Volume2,
  MessageSquare,
  Pencil,
} from "lucide-react";

type IconProps = { size?: number };

/** Puño cerrado (Lucide no tiene uno exacto; SVG del handoff de marca). */
function FistIcon({ size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 13v2.5A3.5 3.5 0 0 0 8.5 19h6a3.5 3.5 0 0 0 3.5-3.5V12" />
      <path d="M5 13a1.6 1.6 0 0 1 3.2 0" />
      <path d="M8.2 12.4a1.6 1.6 0 0 1 3.2 0" />
      <path d="M11.4 12.2a1.6 1.6 0 0 1 3.2 0" />
      <path d="M14.6 12.4a1.6 1.6 0 0 1 3 .3" />
      <path d="M18 12.7c1.1-.5 1-2.4-.5-2.9" />
    </svg>
  );
}

/** Señal de paz (dos dedos en V), en el estilo de línea de Lucide. */
function PeaceIcon({ size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M9.5 12.5 7 4.8a1.7 1.7 0 0 1 3.2-1.1l2 6.3" />
      <path d="M12.2 10 14.8 3a1.7 1.7 0 0 1 3.2 1.2l-2.6 8" />
      <path d="M15.4 12.2c1.8.3 2.6 1.5 2.4 3.3-.3 2.9-2.4 6.5-6.3 6.5-3.1 0-4.5-1.6-5.5-3.6l-1.6-3.5a1.6 1.6 0 0 1 2.9-1.4l.9 1.7" />
    </svg>
  );
}

/** Brazos arriba / persona festejando, en el estilo de línea de Lucide. */
function ArmsUpIcon({ size = 24 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="6" r="2.5" />
      <path d="M12 8.5v6" />
      <path d="M12 10 5.5 4.5" />
      <path d="M12 10l6.5-5.5" />
      <path d="m12 14.5-3 6" />
      <path d="m12 14.5 3 6" />
    </svg>
  );
}

/** Un brazo arriba (persona levantando la mano), estilo de línea Lucide. */
function OneArmUpIcon({ size = 24, mirrored = false }: IconProps & { mirrored?: boolean }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={mirrored ? { transform: "scaleX(-1)" } : undefined}
    >
      <circle cx="12" cy="6" r="2.5" />
      <path d="M12 8.5v6" />
      <path d="M12 10l6.5-5.5" />
      <path d="M12 10l-5 2.5" />
      <path d="m12 14.5-3 6" />
      <path d="m12 14.5 3 6" />
    </svg>
  );
}

/**
 * Emoji (clave en presets/configs) → ícono de línea. Si un emoji no está
 * mapeado se muestra tal cual, así las clases con emojis propios no rompen.
 */
const ICONS: Record<string, (props: IconProps) => ReactElement> = {
  "✋": (p) => <Hand size={p.size} aria-hidden="true" />,
  "✊": (p) => <FistIcon size={p.size} />,
  "👍": (p) => <ThumbsUp size={p.size} aria-hidden="true" />,
  "👎": (p) => <ThumbsDown size={p.size} aria-hidden="true" />,
  "☝️": (p) => <Pointer size={p.size} aria-hidden="true" />,
  "✌️": (p) => <PeaceIcon size={p.size} />,
  "🧍": (p) => <PersonStanding size={p.size} aria-hidden="true" />,
  "🙋": (p) => <OneArmUpIcon size={p.size} />,
  "💪": (p) => <OneArmUpIcon size={p.size} mirrored />,
  "🙌": (p) => <ArmsUpIcon size={p.size} />,
  "😀": (p) => <Smile size={p.size} aria-hidden="true" />,
  "🖼️": (p) => <Image size={p.size} aria-hidden="true" />,
  "🔊": (p) => <Volume2 size={p.size} aria-hidden="true" />,
  "💬": (p) => <MessageSquare size={p.size} aria-hidden="true" />,
  "✏️": (p) => <Pencil size={p.size} aria-hidden="true" />,
};

type GestureIconProps = {
  /** Emoji-clave (p. ej. "✋") o cualquier string; sin match se renderiza tal cual. */
  icon: string;
  size?: number;
  style?: CSSProperties;
};

export default function GestureIcon({ icon, size = 24, style }: GestureIconProps) {
  const render = ICONS[icon.trim()];
  if (!render) {
    return (
      <span style={{ fontSize: size ? `${size}px` : undefined, lineHeight: 1, ...style }}>
        {icon}
      </span>
    );
  }
  return <span style={{ display: "inline-flex", lineHeight: 0, ...style }}>{render({ size })}</span>;
}
