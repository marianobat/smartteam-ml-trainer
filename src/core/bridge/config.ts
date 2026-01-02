const DEFAULT_API_BASE = "https://smartteam-gesture-bridge.marianobat.workers.dev";
const DEFAULT_WS_BASE = "wss://smartteam-gesture-bridge.marianobat.workers.dev/ws";
const DEFAULT_TW_EDITOR = "https://turbowarp.org/editor";
const DEFAULT_EXT_URL = "";
const DEFAULT_TEMPLATE_SB3 = "";

const getEnv = (key: string, fallback: string) => {
  const value = import.meta.env[key] as string | undefined;
  return value && value.trim() ? value.trim() : fallback;
};

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, "");

export const API_BASE = trimTrailingSlash(getEnv("VITE_API_BASE", DEFAULT_API_BASE));
export const WS_BASE = trimTrailingSlash(getEnv("VITE_WS_BASE", DEFAULT_WS_BASE));
export const TW_EDITOR = trimTrailingSlash(getEnv("VITE_TW_EDITOR", DEFAULT_TW_EDITOR));
export const EXT_URL = getEnv("VITE_EXT_URL", DEFAULT_EXT_URL);
export const TEMPLATE_SB3 = getEnv("VITE_TEMPLATE_SB3", DEFAULT_TEMPLATE_SB3);
