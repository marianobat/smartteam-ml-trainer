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
