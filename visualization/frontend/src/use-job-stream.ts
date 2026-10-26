import { useEffect, useRef, useState } from "react";

export type ConnectionState = "idle" | "connecting" | "connected" | "reconnecting" | "unavailable";

/** Share recovery and cleanup semantics between the job list and selected-job stream. */
export function useJobStream<T>(path: string | null, onMessage: (payload: T) => void): ConnectionState {
  const receive = useRef(onMessage);
  receive.current = onMessage;
  const [state, setState] = useState<ConnectionState>("idle");
  useEffect(() => {
    if (!path) {
      setState("idle");
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let delay = 1000;
    setState("connecting");
    const connect = () => {
      if (stopped) return;
      const current = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`);
      socket = current;
      current.onopen = () => {
        if (stopped || socket !== current) return;
        delay = 1000;
        setState("connected");
      };
      current.onmessage = (event) => {
        if (stopped || socket !== current) return;
        try {
          receive.current(JSON.parse(event.data) as T);
        } catch (error) {
          console.warn("Invalid job update", error);
        }
      };
      current.onclose = (event) => {
        if (stopped || socket !== current) return;
        // Authorization failures and missing jobs need user action, not a retry loop.
        if (event.code === 1008) {
          setState("unavailable");
          return;
        }
        setState("reconnecting");
        retry = window.setTimeout(connect, delay);
        delay = Math.min(delay * 2, 15000);
      };
      current.onerror = () => {
        if (!stopped && socket === current) current.close();
      };
    };
    const online = () => {
      if (socket?.readyState !== WebSocket.CLOSED) return;
      window.clearTimeout(retry);
      connect();
    };
    window.addEventListener("online", online);
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(retry);
      window.removeEventListener("online", online);
      socket?.close();
    };
  }, [path]);
  return state;
}
