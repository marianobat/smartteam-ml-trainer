# SmartTEAM ML Trainer (Hands) - MediaPipe + TFJS + kNN

Web app (Vite + React + TypeScript) to capture hand poses, train a classifier, and run live evaluation from the camera. It is an educational tool and a first step toward a future Scratch integration.

---

## What it does (flow)

1) Detects hands with MediaPipe Hands (Tasks Vision) over the camera feed.
2) Converts landmarks into a 10D feature vector that is invariant to position/scale.
3) Lets you capture labeled samples per class (tap/hold).
4) Offers two modes:
   - Por ejemplos (rapido): kNN based on nearest examples.
   - Entrenar un modelo (ML): MLP (TF.js) trained by epochs.
5) Shows live evaluation:
   - Instantaneo (current prediction)
   - Estable (filtered prediction to reduce flicker)

---

## Feature vector (10D)

Instead of raw coordinates, we use finger-curl features for stability:

For each finger (thumb, index, middle, ring, pinky):
1) dist(tip, mcp) / palm size
2) dist(tip, wrist) / palm size

This yields 10 values that describe the pose.

Main file:
- `src/core/hand/featurize.ts`

---

## Classification modes

### 1) Por ejemplos (rapido) - kNN
- No epoch training.
- Train = store examples and classify by euclidean distance (weighted vote).
- Works well with few samples if features are well separated.
- Includes a learning curve by number of samples.

Files:
- `src/core/training/knn.ts`
- `src/core/training/knnCurve.ts`

### 2) Entrenar un modelo (ML) - MLP (TF.js)
- Small neural net with softmax classification.
- Trains by epochs and shows accuracy/loss curves.
- Needs more data, but generalizes better with more variation.

Files:
- `src/core/training/model.ts`
- `src/core/training/train.ts`
- `src/core/training/predict.ts`
- `src/core/training/prepare.ts`

---

## UI layout

Two independent panels with scroll:
- Left: classes, capture, training, learning curve.
- Right: camera + live evaluation (camera stays sticky).

This prevents losing the camera view when there are many classes.

---

## How to use

1) Open the app and accept camera permissions.
2) Create classes (open, fist, index, peace, etc.).
3) Select a class (checkmark button).
4) Capture samples:
   - Tap: 1 sample
   - Hold: after 1s, captures every 0.5s
5) Repeat for all classes (5-30 samples total is ok for quick tests).
6) Choose mode:
   - Por ejemplos (rapido) for instant results with few samples.
   - Entrenar un modelo (ML) if you will collect more data.
7) Click Train.
8) Try live predictions:
   - Instantaneo: direct output
   - Estable: threshold + short confirmation window

---

## Stability notes

The app keeps a stable label to avoid flicker when a pose is ambiguous. This adds a short confirmation window before switching classes. "No hands" resets the internal state so a new pose can appear faster after re-entering the frame.

---

## Project status

MVP working:
- Class management (create/rename/delete/select)
- Sample capture from 2 hands features (10D vector)
- Thumbnails per class
- Training in browser (TFJS) with progress feedback
- Live evaluation with per-class bars + threshold

Pending or future ideas:
- Improve camera/overlay stability on first load
- Tuning for simple gestures with few examples
- Model export (ZIP + manifest)
- Scratch integration (extension/blocks)

---

## Project map

- `src/app/pages/HandTrainer.tsx` - main UI, capture, training, live evaluation.
- `src/core/hand/handLandmarker.ts` - MediaPipe setup + detect loop.
- `src/core/hand/featurize.ts` - feature extraction (10D).
- `src/core/dataset/datasetStore.ts` - dataset state (classes, samples, thumbnails).
- `src/core/training/*` - preparation, training, prediction, kNN.

---

## Documentation

- `docs/AVANCES_Y_PROXIMOS_PASOS.md`

---

## Local development

Requires Node 20.19+ or 22.12+ (see `.nvmrc`).

```bash
npm install
npm run dev
```

Open the URL printed by Vite.

---

## Troubleshooting

- Slow or gradual predictions: adjust stability filters or smoothing.
- ML does not learn with few samples: add more samples per class.
- kNN curve is slow: reduce max steps (default is 20).

---

## License

TBD
