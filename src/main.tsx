import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { startFaviconCycle } from "./app/faviconCycle";

startFaviconCycle();

createRoot(document.getElementById("root")!).render(<App />);
