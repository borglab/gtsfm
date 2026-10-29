import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import type { PhotoState } from "./photo-panel";

export type ViewAxis = "x" | "-x" | "y" | "-y" | "z" | "-z";
export interface ViewerState {
  busy: boolean;
  mode: "idle" | "scene" | "splat";
  background: string;
  orthographic: boolean;
  showGround: boolean;
  groundY: number;
  showCams: boolean;
  pointSize: number;
  statsVisible: boolean;
  points: number;
  cameras: number;
  splats: number;
  imageName: string;
  aligned: boolean;
  alignmentMessage: string;
  loading: { active: boolean; message: string; progress: number; compact: boolean };
}

/** The UI depends on this interface, not Babylon meshes or DOM selectors. */
export interface ViewerApi {
  getState(): ViewerState;
  subscribe(listener: (state: ViewerState) => void): () => void;
  dispose(): void;
  onFirstFrame(listener: () => void): void;
  isBusy(): boolean;
  getPhotoState(): PhotoState;
  subscribePhotos(listener: (state: PhotoState) => void): () => void;
  selectCamera(index: number): void;
  loadScene(input: {
    pointsUrl: string;
    imagesUrl?: string | null;
    camerasUrl?: string | null;
    scenePath?: string;
    splatsUrl?: string | null;
  }): Promise<boolean | void>;
  loadSplatsFile(input: { splatsUrl: string; label: string; streamId?: string }): Promise<boolean | void>;
  frameScene(): void;
  toggleOrthographicProjection(): void;
  viewFromAxis(axis: ViewAxis): void;
  alignZUp(): void;
  setBackground(theme: string): void;
  setGroundY(value: number): number;
  toggleGroundVisible(): boolean;
  setShowCams(value: boolean): void;
  setPointSize(value: number): void;
  toggleStatsVisible(): void;
  prevCam(): void;
  nextCam(): void;
  savePreferences(): void;
}

declare global {
  interface Window {
    createGtsfmViewer?: (canvas: HTMLCanvasElement) => ViewerApi;
    /** Kept for debugging; UI components use the context. */
    gtsfmViewer?: ViewerApi;
  }
}

const INITIAL_STATE: ViewerState = {
  busy: false,
  mode: "idle",
  background: "dark",
  orthographic: false,
  showGround: false,
  groundY: 0,
  showCams: true,
  pointSize: 2,
  statsVisible: true,
  points: 0,
  cameras: 0,
  splats: 0,
  imageName: "",
  aligned: false,
  alignmentMessage: "",
  loading: { active: false, message: "", progress: 0, compact: false },
};
const ViewerContext = createContext({
  viewer: null as ViewerApi | null,
  setViewer: (_viewer: ViewerApi | null) => {},
});

const ViewerStateContext = createContext(INITIAL_STATE);

export function ViewerProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewer] = useState<ViewerApi | null>(null);
  const [state, setState] = useState(INITIAL_STATE);
  useEffect(() => {
    if (!viewer) {
      setState(INITIAL_STATE);
      return;
    }
    const unsubscribe = viewer.subscribe(setState);
    setState(viewer.getState());
    return unsubscribe;
  }, [viewer]);
  const api = useMemo(() => ({ viewer, setViewer }), [viewer]);
  return (
    <ViewerContext.Provider value={api}>
      <ViewerStateContext.Provider value={state}>{children}</ViewerStateContext.Provider>
    </ViewerContext.Provider>
  );
}

export const useViewerApi = () => useContext(ViewerContext);
export const useViewer = () => ({ ...useViewerApi(), state: useContext(ViewerStateContext) });

/** Results only need busy transitions, not every streamed point-count update. */
export function useViewerBusy() {
  const { viewer } = useViewerApi();
  const subscribe = useCallback((notify: () => void) => viewer?.subscribe(notify) ?? (() => {}), [viewer]);
  const snapshot = useCallback(() => viewer?.isBusy() ?? false, [viewer]);
  return useSyncExternalStore(subscribe, snapshot);
}

/** Mount independently of API requests; release the complete renderer on unmount. */
export function ViewerCanvas() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { setViewer } = useViewerApi();
  useEffect(() => {
    let instance: ViewerApi | null = null;
    const start = () => {
      if (instance || !canvas.current || !window.createGtsfmViewer) return;
      try {
        instance = window.createGtsfmViewer(canvas.current);
        instance.onFirstFrame(() => window.dispatchEvent(new Event("gtsfm-first-frame")));
        window.gtsfmViewer = instance;
        setViewer(instance);
      } catch (error) {
        window.dispatchEvent(new CustomEvent("gtsfm-startup-error", { detail: String(error) }));
      }
    };
    window.addEventListener("gtsfm-engine-ready", start);
    start();
    return () => {
      window.removeEventListener("gtsfm-engine-ready", start);
      instance?.dispose();
      if (window.gtsfmViewer === instance) delete window.gtsfmViewer;
      setViewer(null);
    };
  }, [setViewer]);
  return <canvas id="renderCanvas" ref={canvas} />;
}
