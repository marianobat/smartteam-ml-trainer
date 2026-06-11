# Plan de mejora UX — SmartTEAM ML Trainer (estilo LEGO Coding Canvas)

> **Cómo usar este documento con Cursor:** ejecutá las 7 etapas EN ORDEN, una por vez.
> Para cada etapa, pegá en Cursor: el bloque "Contexto para cada prompt", las secciones
> "Sistema de diseño" / "Arquitectura" / "Cambios de lógica" como referencia, y el texto
> de la etapa. Verificá los criterios de aceptación en el navegador antes de pasar a la
> siguiente. ⚠️ ANTES de la Etapa 3: exportá un ZIP de un proyecto real con la versión
> actual de la app — es el fixture para verificar la migración v1→v2 al final.

## Contexto

La app funciona completa (6 modalidades ML, TurboWarp, micro:bit a pedido, persistencia IndexedDB/ZIP, PiP) pero su interfaz es de desarrollador: estilos inline, jerga técnica visible ("época", "validación", "umbral", "subscribers"), miniaturas como foto JPEG cruda y sin guía de flujo. La referencia elegida es **LEGO Coding Canvas** (capturas analizadas): cámara protagonista con esqueleto grueso multicolor sobre video atenuado, clases como tarjetas visuales con dibujo de pose (no foto), chip de progreso "4/5" con mínimo de muestras, botón Entrenar bloqueado hasta cumplirlo, barras de evaluación grandes con %, indicador de pasos.

**Decisiones tomadas con el usuario (no reabrir):**
1. Alcance: **todo de una** — Home, selector y los 3 entrenadores (video cubre manos/cara/cuerpo/imágenes).
2. Lo técnico (curvas, KNN/ML, WebSocket, umbral, proyecto) → **toggle "modo avanzado"** (⚙️, localStorage).
3. Estética: **paleta vibrante infantil nueva** (reemplaza naranja/teal), tipografía Fredoka (display) + Nunito (cuerpo).
4. Stack: **CSS plano + design tokens** (`src/theme.css` + un .css por componente). Sin Tailwind ni dependencias nuevas.
5. Miniaturas manos/cuerpo/cara: **solo esqueleto dibujado** sobre blanco (privacidad — no se guarda foto del menor). Imágenes mantiene foto.
6. Flujo: **pasos visibles en una sola pantalla** (① Enseñale ejemplos → ② Entrená → ③ Probalo y conectalo), se marcan solos, sin bloqueos.
7. **Mínimo 5 muestras por clase** con chip "3/5" y placeholders punteados; Entrenar se habilita al completar.

**Hallazgos del análisis de código que condicionan el plan:**
- Las miniaturas hoy NO están ligadas a las muestras (`ADD_SAMPLE` y `ADD_THUMBNAIL` separadas, `thumbnailsByClass` cap 20, sin borrado individual). El tachito por muestra exige `Sample.id` + `thumb` + `REMOVE_SAMPLE` → **cambio de modelo de datos + migración de persistencia v1→v2**.
- `imageExtractor.processFrame` no dibuja nada en el canvas → la fuente de miniatura debe ser configurable (`thumbnailSource: "overlay" | "video"` en `TrainerConfig`).
- `missingLabel` viaja por WS y micro:bit: **no renombrar**; el texto amigable es un campo nuevo `missingHint` solo de UI.
- AudioTrainer no usa `datasetStore` (speech-commands maneja los ejemplos): el reuso ahí es solo presentacional; sin borrado individual de audio.

---

## Sistema de diseño (`src/theme.css`, nuevo)

```css
@import url("https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700;800&display=swap");
:root {
  /* Roles */
  --color-primary: #7C4DFF;      --color-primary-soft: #EDE7FF;
  --color-secondary: #00BCD9;    --color-accent: #FF4D8D;
  --color-success: #2EC56B;      --color-warning: #FFC838;
  --color-danger: #FF5A5A;
  --color-ink: #2D2A45;          --color-ink-soft: rgba(45,42,69,.65);
  --color-bg: #F6F4FF;           --color-surface: #FFF;
  --color-outline: rgba(45,42,69,.12);
  /* Color por modalidad (cards selector + acento del trainer) */
  --mod-hands:#FF4D8D; --mod-face:#7C4DFF; --mod-pose:#00BCD9;
  --mod-images:#FF8A3D; --mod-text:#4D7CFE; --mod-audio:#2EC56B;
  /* Tipos y geometría */
  --font-display:"Fredoka","Nunito",system-ui,sans-serif;
  --font-body:"Nunito",system-ui,sans-serif;
  --radius-sm:10px; --radius-md:16px; --radius-lg:24px; --radius-pill:999px;
  --shadow-card:0 6px 20px rgba(70,50,140,.10); --shadow-pop:0 12px 32px rgba(70,50,140,.18);
  --tap-min:44px; --tap-big:72px;
}
```

Esqueleto (canvas no lee CSS vars a 60fps) → constantes TS en `src/core/overlay/skeletonStyle.ts`: rosa `#FF4D8D`, cian `#22D3EE`, violeta `#A855F7`, `lineWidth: 14`, joints blancos radio 9, `lineCap/lineJoin: "round"`.

## Arquitectura de componentes nuevos

Carpeta `src/app/components/trainer/` — **presentacionales, props planas** (no conocen datasetStore ni TF), reusables por los 3 entrenadores: `StepsBar`, `ClassCardStrip` (fila de tarjetas de clase + "Agregar"), `SampleGrid` (miniaturas + tachito + placeholders punteados), `CameraStage` (video atenuado + canvas + hint), `CaptureControls` (botón verde 72px + toggle foto/ráfaga, overlay sobre cámara), `TrainPanel`, `LivePredictionBars`, `StatusChips` (guardado ✓ / TurboWarp / micro:bit), `AdvancedDrawer`. Soporte: `src/app/hooks/useAdvancedMode.ts` (localStorage `st.advancedMode`), `src/app/copy.ts` (todo el copy del modo chico centralizado), `Trainer.css` (grid `1.5fr | minmax(340px,1fr)`, colapsa a 1 col <1100px y elimina el estado `isNarrow` de JS).

**Visibilidad**: chico ve pasos, cámara, clases, muestras, Entrenar, barras en vivo, chips de estado, "Conectar micro:bit" compacto y el botón PiP. Drawer avanzado: toggle KNN/ML, recharts (montar solo con drawer abierto), Precision/Validación/Época, panel WebSocket completo, umbral + log micro:bit, ProjectPanel completo.

**Copy** (tabla completa en `src/app/copy.ts`): "Capturar muestra" → "📸 ¡Sacar foto!"; "Entrenar" → "✨ ¡Entrenar modelo!"; "Entrenando (epoca 12/40)" → "Aprendiendo... 🧠" + barra; "Sin manos" (UI) → "No veo tus manos 👀" (`missingHint`); "Instantáneo/Estable/aceptado" → "Veo: **Hola** 99%"; "Publicador WebSocket" → chip "TurboWarp ✓"; "Muestras: N" → "N ejemplos" + chip "N/5".

## Cambios de lógica (los únicos en `core/`)

1. **`datasetStore.ts`**: `Sample` gana `id`, `thumb?`, `note?`; `ADD_SAMPLE` acepta thumb/note (reemplaza el par ADD_SAMPLE+ADD_THUMBNAIL); nueva `REMOVE_SAMPLE`; `MIN_SAMPLES_PER_CLASS = 5`; `thumbnailsByClass` queda deprecado solo-lectura.
2. **`projectStore.ts` + `projectZip.ts`**: `version: 2` + `migrateDatasetV1()` best-effort (asigna ids, reparte thumbs viejas — fotos — sobre las muestras más recientes; aceptan v1 al cargar/importar).
3. **Miniatura esqueleto**: `captureSkeletonThumbnail(overlayCanvas)` rasteriza el canvas de overlay sobre fondo blanco (PNG 96px, modo contain, espejado); válido porque `captureSample` solo corre con sujeto detectado (el overlay tiene el esqueleto del mismo frame). `TrainerConfig.thumbnailSource: "overlay" | "video"` ("video" solo en imágenes).
4. **`canTrain`** nuevo: `classes.length >= 2 && todas con count >= 5`.
5. **Overlays**: draw.ts (mano izq rosa / der cian, grosor 14, joints blancos), drawPose (multicolor por grupo: torso violeta, brazos rosa/cian), drawFace (grosor 6, violeta). Video atenuado vía CSS `filter: saturate(.35) brightness(.8)` — NO aplicar en modalidad imágenes.
6. **Pasos derivados**: paso1 = todas las clases ≥5; paso2 = trainComplete; paso3 = se latchea con la primera predicción aceptada.

---

## INDICACIONES PARA CURSOR — 7 etapas (pegar de a una, verificar antes de seguir)

> Contexto para cada prompt: repo `smartteam-ml-trainer`, React 19 + Vite + TS, CSS plano. Regla de oro: NO tocar la lógica de entrenamiento (`core/training`), bridge WS (`core/bridge`), micro:bit (`core/microbit`), PiP, salvo lo indicado. `missingLabel` no se renombra (viaja por WS/serial).

### Etapa 1 — Sistema de diseño base
Crear `src/theme.css` con los tokens de arriba (paleta, modalidades, Fredoka/Nunito, radios, sombras, --tap-min 44px). Reescribir `src/index.css`: importar theme.css, body con `--color-bg` y `--font-body`, títulos con `--font-display`, estilos base de button/input con altura mínima `--tap-min`, `:focus-visible` con anillo violeta. No tocar ningún `.tsx` ni `core/`.
**Aceptación**: la app entera cambia a paleta lavanda/violeta y tipografías nuevas; todo funciona igual (Home, selector, los entrenadores cargan y entrenan).

### Etapa 2 — Esqueletos protagonistas + cámara cine
Crear `src/core/overlay/skeletonStyle.ts` (constantes SKEL de arriba). Editar `src/core/hand/draw.ts` (izq rosa / der cian, lineWidth 14, lineCap/lineJoin round, joints círculo blanco radio 9 con borde de color), `drawPose` en `src/core/extractors/poseExtractor.ts` (grosor 14, multicolor por grupos de POSE_CONNECTIONS: cara violeta índices <11, brazos 11-22 rosa/cian por lado, piernas 23+ rosa/cian), `drawFace` en `faceExtractor.ts` (grosor 6, violeta). En `Trainer.tsx` aplicar al `<video>` filtro inline `saturate(0.35) brightness(0.8)` SOLO si `extractor.id !== "image"`. No tocar `featurize*` ni `processFrame` ni el contrato `VideoExtractor`.
**Aceptación**: en manos/cuerpo/cara el video se ve atenuado con esqueleto grueso rosa/cian/violeta y joints blancos; en imágenes video normal; detección y predicción intactas.

### Etapa 3 — Producto: muestras con id, miniatura-esqueleto, mínimo 5, borrado
Editar `src/core/dataset/datasetStore.ts`: `Sample` += `id: string`, `thumb?: string`, `note?: string`; `ADD_SAMPLE` acepta thumb/note; acción nueva `REMOVE_SAMPLE {id}`; exportar `MIN_SAMPLES_PER_CLASS = 5`; deprecar `ADD_THUMBNAIL`/`thumbnailsByClass` (mantener en el tipo para leer proyectos viejos). Editar `src/core/storage/projectStore.ts` y `src/core/export/projectZip.ts`: `version: 2`, aceptar v1 con `migrateDatasetV1()` (ids nuevos + repartir `thumbnailsByClass` newest-first sobre las muestras más recientes por clase). Crear `src/app/components/trainer/thumbnails.ts` con `captureSkeletonThumbnail(overlay, size=96, mirror=true)` (canvas blanco, contain, PNG). En `Trainer.tsx`: `captureSample` despacha UNA acción con thumb (overlay o video según `TrainerConfig.thumbnailSource` nuevo — "video" solo en images, setear en `TrainerPage.tsx`); `canTrain` nuevo (≥2 clases, todas ≥5); render provisional de thumbs desde `sample.thumb` + botón borrar por muestra. En `TextTrainer.tsx`: `canTrain` nuevo + guardar el texto en `note` (preparando eliminar `textsByClass` en Etapa 6).
**Aceptación**: capturas en manos/cuerpo/cara generan miniatura de esqueleto sobre blanco; en imágenes foto; Entrenar deshabilitado hasta 5/5 en todas; borrar una muestra baja el contador; recargar conserva todo; un ZIP exportado ANTES de esta etapa importa sin romper (fotos viejas se muestran).

### Etapa 4 — Rediseño del Trainer de video
Crear en `src/app/components/trainer/`: `StepsBar`, `ClassCardStrip`, `SampleGrid`, `CameraStage`, `CaptureControls`, `StatusChips` (cada uno .tsx + .css, presentacionales con props planas) y `Trainer.css`. Reescribir el JSX de `src/app/pages/Trainer.tsx` con el layout: header (← Volver, título Fredoka, StatusChips, ⚙️ placeholder), StepsBar, grid `1.5fr` cámara | `1fr` panel derecho (ClassCardStrip horizontal con miniatura + "N ejemplos" + chip amarillo "3/5" + tarjeta ➕; SampleGrid con tachito al hover y placeholders punteados hasta 5; botón Entrenar). CaptureControls superpuesto sobre la cámara: botón redondo verde 72px (mantener startHold/endHold) + toggle 📷 una / 🎞️ ráfaga (estado nuevo `burstMode`; en "una" se ignoran los timers de hold). Mover el filtro del video a `CameraStage.css`. Eliminar `isNarrow` (media query <1100px → 1 columna). Los paneles técnicos viejos quedan apilados abajo sin estilo (se ordenan en Etapa 5). NO tocar hooks/refs/loop/WS/persistencia: solo JSX y estilos.
**Aceptación**: cámara grande con botón verde superpuesto y toggle ráfaga; tarjetas de clase con esqueleto-miniatura; grilla con placeholders; StepsBar se marca sola; responsive a 1 columna; entrenar/predicción/WS/micro:bit/PiP siguen funcionando.

### Etapa 5 — Modo avanzado + evaluación en vivo + copy
Crear `src/app/hooks/useAdvancedMode.ts` (localStorage `st.advancedMode`, default false), `AdvancedDrawer`, `LivePredictionBars` (barras grandes con % y color de clase), `TrainPanel` ("✨ ¡Entrenar modelo!" / "Aprendiendo... 🧠" con barra de progreso), `src/app/copy.ts` con TODO el copy del modo chico (tabla del plan). En `Trainer.tsx`: mover al drawer el toggle examples/ML, recharts (montar solo con drawer abierto), panel WebSocket, ProjectPanel y detalles de micro:bit; `MicrobitPanel.tsx` gana prop `advanced: boolean` (compacto: botón conectar + estado + pedidos; completo: + umbral + log). Reemplazar "Instantáneo/Estable/aceptado/pendiente/Umbral" por "Veo: **X** 99%". Agregar `missingHint` a `TrainerConfig` ("No veo tus manos 👀", etc.) sin tocar `missingLabel`.
**Aceptación**: con ⚙️ apagado no se ve ninguna jerga; barras en vivo grandes con %; el drawer muestra todo lo técnico funcionando; el toggle sobrevive recargas; conectar micro:bit accesible sin modo avanzado.

### Etapa 6 — TextTrainer y AudioTrainer
`TextTrainer.tsx`: reusar StepsBar/ClassCardStrip/SampleGrid (`content` = snippet desde `sample.note`)/TrainPanel/LivePredictionBars/AdvancedDrawer + copy; eliminar `textsByClass` (migrado a `note` en Etapa 3); borrado individual funciona. `AudioTrainer.tsx`: mismos componentes presentacionales; miniatura = ícono de onda SVG; subir mínimo 3→5 (incluida "Ruido de fondo", reusar el chip "N/5"); CaptureControls variante "🎙 grabar 1s"; sin borrado individual (speech-commands no lo facilita — dejar comentario). No tocar `core/text/` ni la lógica de speech-commands.
**Aceptación**: ambos entrenadores estructuralmente idénticos al de video (pasos, tarjetas, chips, drawer); textos borra muestras individuales; audio exige 5 por clase; entrenar y vivo funcionan.

### Etapa 7 — Home + selector + accesibilidad + QA
`Home.tsx/.css`: hero lúdico, card Entrenador protagonista, sesión TurboWarp como card secundaria ("Para conectar con TurboWarp"). `TrainerPage.tsx/.css`: cards con `--mod-*`, íconos grandes por modalidad, warning de sesión con el copy nuevo. Pasada AA en todos los componentes nuevos: texto sobre amarillo/verde siempre `--color-ink` (contraste ≥4.5:1), foco visible por teclado en todo control, `aria-pressed` en toggles, respetar `prefers-reduced-motion`.
**Aceptación / regresión integral**: navegación completa Home → selector → cada una de las 6 modalidades → entrenar → probar; export/import ZIP (incluido uno v1 viejo); recarga con persistencia; PiP; micro:bit con hardware; publicación WS con room creada.

---

## Riesgos clave

1. **Persistencia v1→v2** (el mayor): exportar un ZIP con la versión actual ANTES de empezar y verificar al final que importa. La migración mantiene las fotos viejas como thumbs.
2. Proyectos guardados con <5 muestras/clase quedan con Entrenar deshabilitado hasta sumar — deseado, los placeholders lo comunican.
3. Imágenes: mantener foto y NO atenuar su video (condicionar por `extractor.id !== "image"`).
4. Tamaño IndexedDB: thumb por muestra → PNG 96px; opcional omitir thumb pasada la muestra ~60 por clase.
5. Recharts solo montado con drawer abierto (performance en Chromebooks).
6. Contraste: `#FFC838` y `#2EC56B` no soportan texto blanco — siempre `--color-ink`.

## Verificación global

- `npm run build` y `npm run lint` limpios al final de cada etapa.
- Tras Etapa 3 y al final: ciclo completo en texto (entrenar → recargar → restaurado → predice) que es verificable sin cámara/mic.
- Prueba física final del usuario: cámara en manos/cuerpo/cara/imágenes, micrófono en audio, micro:bit, y prueba con un chico real (UX-4 del plan original).
