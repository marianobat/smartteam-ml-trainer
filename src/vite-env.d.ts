/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENABLE_TURBOWARP?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
