import { useState } from "react";
import HandTrainer from "./HandTrainer";

export default function Home() {
  const [mode, setMode] = useState<"home" | "hand">("home");

  if (mode === "hand") return <HandTrainer onBack={() => setMode("home")} />;

  return (
    <div style={{ padding: 16 }}>
      <h1>SmartTEAM Trainer</h1>
      <p>Elegí el tipo de modelo.</p>
      <button onClick={() => setMode("hand")}>Mano (2 manos)</button>
    </div>
  );
}