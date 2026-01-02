import { useMemo } from "react";
import HandTrainer from "./HandTrainer";
import { getToken, getRoom } from "../../core/bridge/session";

const getRoomFromQuery = () => {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  return params.get("room") ?? "";
};

export default function TrainerPage() {
  const baseUrl = useMemo(() => import.meta.env.BASE_URL ?? "/", []);
  const room = useMemo(() => getRoomFromQuery() || getRoom() || "", []);
  const publishToken = useMemo(() => getToken() || "", []);

  return (
    <HandTrainer
      onBack={() => window.location.assign(baseUrl)}
      room={room}
      publishToken={publishToken}
    />
  );
}
