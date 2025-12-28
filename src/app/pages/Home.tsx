import { Suspense, lazy, useState } from "react";

const HandTrainer = lazy(() => import("./HandTrainer"));

export default function Home() {
  const [mode, setMode] = useState<"home" | "hand">("home");

  if (mode === "hand") {
    return (
      <Suspense fallback={<div style={{ padding: 16 }}>Cargando entrenador…</div>}>
        <HandTrainer onBack={() => setMode("home")} />
      </Suspense>
    );
  }

  return (
    <div style={{ padding: 16 }}>
      <h1>SmartTEAM Trainer</h1>
      <p>Elegí el tipo de modelo.</p>
      <button onClick={() => setMode("hand")}>Mano (2 manos)</button>
    </div>
  );
}
