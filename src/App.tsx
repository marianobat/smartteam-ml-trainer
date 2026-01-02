import Home from "./app/pages/Home";
import Program from "./app/pages/Program";
import TrainerPage from "./app/pages/TrainerPage";

const getRoute = () => {
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const base = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
  let path = window.location.pathname;
  if (base && path.startsWith(base)) {
    path = path.slice(base.length);
  }
  const normalized = path.replace(/\/+$/, "") || "/";
  if (normalized.endsWith("/trainer")) return "trainer";
  if (normalized.endsWith("/program")) return "program";
  return "home";
};

export default function App() {
  const route = getRoute();
  if (route === "trainer") {
    return <TrainerPage />;
  }
  if (route === "program") {
    return <Program />;
  }
  return <Home />;
}
