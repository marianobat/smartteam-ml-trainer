# SmartTEAM ML Trainer — Avances y próximos pasos

## Contexto
Este repo implementa un primer MVP de entrenamiento y testeo en navegador (tipo Teachable Machine) para reconocer gestos de manos (2 manos) y preparar el camino hacia integración posterior con Scratch (como extensión / bloques).

El objetivo pedagógico es que estudiantes puedan:
- Crear clases (ej: OPEN / FIST)
- Capturar ejemplos
- Entrenar un modelo
- Ver la evaluación en vivo (probabilidades por clase)
- Ajustar dataset hasta lograr una detección estable

---

## Avances (MVP logrado)

### 1) Captura y dataset
- UI para gestionar clases: crear, renombrar, eliminar, seleccionar clase activa.
- Captura de muestras a partir del vector de features (128 dims) generado desde landmarks de MediaPipe Hands (2 manos).
- Validación básica para no capturar frames “sin manos” (flags left/right).
- Miniaturas por clase (thumbnails) para “ver qué estoy entrenando”.
- Límite configurable de miniaturas por clase (para evitar exceso de memoria/UI lenta).

### 2) Entrenamiento en navegador (TFJS)
- Pipeline dataset → tensores:
  - xs shape [N,128]
  - ys one-hot shape [N, numClasses]
- Entrenamiento de clasificador MLP:
  - Dense(32) + Dropout(0.3) + Softmax (numClasses)
- UX de entrenamiento:
  - feedback de “entrenando”
  - progreso por epoch
  - gráfico de accuracy (train y val), con regla de UX: no enfatizar val acc con datasets muy chicos.
- Regla: ocultar/omitir val acc hasta contar con un volumen mínimo razonable (>=60 samples totales).

### 3) Evaluación en vivo (Live Test)
- Predicción en vivo con TFJS:
  - throttle (p. ej. cada 200ms) para performance estable.
  - smoothing (EMA) para evitar parpadeo de clases.
- Visualización por barras horizontales fijas:
  - 1 barra por clase (orden fijo)
  - barra gris hasta superar threshold (0.7), luego verde.

### 4) Normalización (obligatoria)
- Normalización aplicada en train y predict para mejorar estabilidad:
  - restar wrist por mano y escalar (wrist → middle_mcp), manteniendo output final en 128 dims.

---

## Lo que falta / pendientes conocidos

### A) Estabilidad “a la primera”
- En algunos escenarios, el overlay o la detección puede requerir re-entrada (issue típico de cámara + montaje en dev/StrictMode).
- Pendiente: robustecer inicialización y cleanup para asegurar render/detección consistente sin re-entrar.

### B) Afinado de detección “coarse”
- Para gestos simples (puño vs abierto), aún puede requerir ajuste fino:
  - calidad de muestras (variabilidad)
  - threshold / smoothing
  - normalización y/o features derivados (ej: curl por dedo)

### C) UX pedagógica
- Simplificar mensajes, mínimos recomendados y guías dentro de la UI:
  - sugerir “8–15 muestras por clase” como punto de partida
  - si no logra estabilidad, recomendar sumar 2–4 ejemplos más por clase

---

## Próximos pasos (orden sugerido)

### 1) Consolidar estabilidad y performance (core)
- Asegurar que cámara + canvas + loop se inicialicen bien en el primer ingreso.
- Asegurar cleanup total al cambiar de pantalla:
  - detener stream de cámara
  - cancelar RAF
  - limpiar timers
- Revisar mirror/handedness (izquierda/derecha) para consistencia visual.

### 2) Mejoras de aprendizaje (manteniendo simple para alumnos)
- Ajustar “coarse detection”:
  - threshold dinámico y/o debounce por tiempo (ej: 2 ticks consecutivos)
  - ajuste de EMA (alpha)
- Opcional (si MLP sigue inestable con pocos datos):
  - modo alternativo “Prototype Classifier” (centroid/cosine) como fallback pedagógico.
- Mostrar recomendaciones en UI según tamaño del dataset:
  - si N muy bajo, no mostrar val acc o no dividir train/val.

### 3) Exportación (ZIP) del modelo entrenado
- Exportar un zip con:
  - `smartteam-model.json` (manifest)
  - `labels.json`
  - `model/model.json` + `weights.bin`
  - metadata: threshold recomendado, normalización usada, versión
- (Más adelante) importar un zip para reusar modelos.

### 4) Integración con Scratch (fase siguiente)
- Definir formato del modelo/manifest compatible con la extensión.
- Construir una extensión Scratch que:
  - cargue el zip
  - corra inferencia en el navegador
  - exponga bloques tipo:
    - “predicción actual”
    - “confianza de clase X”
    - “cuando clase == X (evento)”
- Integrar en una versión propia de Scratch (Stretch3 / scratch-gui custom) embebida en el sitio SmartTEAM.

### 5) Producto / packaging
- Branding (logo, dominio, landing simple).
- Documentación para docentes:
  - guía de actividades
  - ejemplos de gestos por clase
  - checklist de entrenamiento
- Preparar roadmap para sumar otros tipos de modelos:
  - imagen, pose cuerpo, cara, texto (futuro)

---

## Cómo correr local
```bash
npm install
npm run dev
