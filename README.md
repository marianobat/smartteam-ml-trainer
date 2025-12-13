# SmartTEAM ML Trainer (Hands) — MVP

Un MVP estilo **Teachable Machine** para **entrenar y probar en el navegador** un clasificador de gestos de manos (2 manos) usando **MediaPipe Hands + TensorFlow.js**.  
Este trainer es el primer paso hacia una integración posterior con **Scratch** (extensión/bloques).

> Objetivo pedagógico: que estudiantes creen clases (ej. “OPEN”, “FIST”), capturen ejemplos, entrenen y vean la predicción en vivo.

---

## Estado del proyecto

✅ MVP funcionando:
- Gestión de clases (crear / renombrar / eliminar / seleccionar)
- Captura de muestras desde features de 2 manos (vector 128)
- Miniaturas por clase (thumbnails) para visualizar qué se está entrenando
- Entrenamiento en navegador (TFJS) con feedback de progreso
- Evaluación en vivo con barras horizontales por clase + threshold
- Normalización obligatoria de features (train + predict)

⚠️ Pendiente de afinado:
- Estabilidad de cámara/overlay en el primer ingreso en algunos casos
- Ajustes “coarse detection” para que gestos simples funcionen mejor con pocos ejemplos

---

## Requisitos
- Node.js recomendado: **20+**
- npm

---

## Instalación y ejecución local

```bash
npm install
npm run dev

Abrí la URL que imprime Vite (por ejemplo http://localhost:5173).

⸻

## Cómo usar (flujo sugerido)

1. Entrá a **Hand Trainer (2 manos)**.
2. Creá 2 clases (ej. `OPEN` y `FIST`).
3. Seleccioná una clase y capturá ejemplos (tap o “press & hold”, según el modo actual).
4. Repetí para la otra clase.
5. Hacé click en **Train**.
6. Probá la predicción en vivo mirando las barras por clase y el estado del threshold.

### Recomendación de muestras (pedagógico)

- Punto de partida: **8–15 muestras por clase**, bien distintas.
- Si no supera el threshold de manera estable, sumar **2–4** muestras más por clase.

---

## Documentación

- 📌 Avances y próximos pasos: `docs/AVANCES_Y_PROXIMOS_PASOS.md`

---

## Estructura (alto nivel)

- `src/app/pages/HandTrainer.tsx`  
  UI del trainer + captura + entrenamiento + evaluación en vivo

- `src/core/dataset/`  
  Store/reducer de clases, samples y miniaturas

- `src/core/hand/`  
  HandLandmarker, featurización, dibujo, normalización

- `src/core/training/`  
  Preparación de tensores, modelo, entrenamiento, predicción

---

## Roadmap (resumen)

1) Mejorar estabilidad de cámara/overlay al primer ingreso  
2) Afinar detección “coarse” (threshold, smoothing, decisión estable)  
3) Exportación de modelo (ZIP + manifest)  
4) Integración con Scratch (extensión/bloques)  
5) Ampliación a otros modelos: imagen, pose cuerpo, cara, texto

---

## Licencia / atribución

El proyecto está pensado para mantenerse **abierto** y reconocer el origen de las tecnologías utilizadas (MediaPipe / TFJS).  
La comercialización del proyecto SmartTEAM se apoya en **libros para el aula + acompañamiento pedagógico**.
