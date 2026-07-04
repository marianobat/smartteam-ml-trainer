// src/app/copy.ts
//
// Todo el texto visible del "modo chico" (8-14 años), centralizado.
// Voz: español latino neutro (tuteo), cálida y en segunda persona.
// Regla: nada de jerga (época, validación, umbral, WebSocket) fuera del
// modo avanzado. Sin emojis: los íconos los ponen los componentes (Lucide).

export const COPY = {
  back: "← Volver",
  advanced: "Modo avanzado",

  steps: ["Enséñale ejemplos", "Entrena", "Pruébalo y conéctalo"] as const,

  addClass: "Agregar",
  className: "Nombre de la clase",
  deleteClass: "Eliminar clase",
  deleteSample: "Borrar ejemplo",
  examplesCount: (n: number) => `${n} ejemplo${n === 1 ? "" : "s"}`,

  captureHint: "Mantén presionado el botón para tomar muchas seguidas",
  captureOne: "Una por una",
  captureBurst: "Ráfaga",
  recordAudio: "Grabar 1 segundo",

  train: "¡Entrenar modelo!",
  training: "Aprendiendo...",
  trained: "¡Tu modelo ya aprendió! Pruébalo",
  needTwoClasses: "Crea al menos 2 clases para poder entrenar",
  needSamples: (min: number) => `Cada clase necesita ${min} ejemplos para entrenar`,

  tryTitle: "Pruébalo",
  see: "Veo:",
  seeNothing: "No estoy seguro todavía...",
  liveEmpty: "Cuando entrenes tu modelo, aquí vas a ver qué detecta en vivo.",
  pipOpen: "Ventana flotante",
  pipClose: "Cerrar ventana",

  chipSaved: "Guardado",
  chipSaving: "Guardando...",
  chipTurboWarp: "TurboWarp",
  chipMicrobit: "micro:bit",

  testTextPlaceholder: "Escribe algo y mira qué clase detecta...",
  addTextPlaceholder: "Escribe un ejemplo para esta clase y presiona Enter...",
  addTextButton: "Agregar ejemplo",
};
