import type { ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useViewer, type ViewAxis } from "./viewer-context";

export function ViewControls() {
  const { viewer, state } = useViewer();
  return (
    <div id="viewControls" role="group" aria-label="Projection and axis views">
      <button
        id="orthographicBtn"
        type="button"
        disabled={!viewer}
        className={state.orthographic ? "is-active" : ""}
        title="Toggle orthographic projection (parallel lines stay parallel)"
        aria-pressed={state.orthographic}
        onClick={() => {
          viewer?.toggleOrthographicProjection();
          viewer?.savePreferences();
        }}
      >
        Orthographic
      </button>
      <button
        id="alignZUpBtn"
        className="hud-scene-only"
        disabled={!viewer || state.busy}
        title="Fit a plane to camera centers and rotate its normal to Z-up"
        aria-pressed={state.aligned}
        onClick={() => viewer?.alignZUp()}
      >
        {state.aligned ? "Reset Z-up" : "Align Z-up"}
      </button>
      <span id="alignmentStatus" className="hud-scene-only" role="status" hidden={!state.alignmentMessage}>
        {state.alignmentMessage}
      </span>
      <div className="axis-view-controls" role="group" aria-label="Axis views">
        {(["x", "-x", "y", "-y", "z", "-z"] as ViewAxis[]).map((axis) => {
          const label = axis.startsWith("-") ? `−${axis.slice(1).toUpperCase()}` : `+${axis.toUpperCase()}`;
          return (
            <button
              key={axis}
              type="button"
              data-view-axis={axis}
              disabled={!viewer}
              title={`View from ${label} toward the current center`}
              aria-label={`View from ${label}`}
              onClick={() => viewer?.viewFromAxis(axis)}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ViewerToolbar() {
  const { viewer, state } = useViewer();
  return (
    <div id="hud" className={state.mode === "splat" ? "hud-splat-mode" : ""}>
      <label className="background-control">
        BG{" "}
        <select
          id="backgroundSelect"
          value={state.background}
          disabled={!viewer}
          aria-label="Viewer background"
          onChange={(event) => {
            viewer?.setBackground(event.target.value);
            viewer?.savePreferences();
          }}
        >
          <option value="dark">Dark</option>
          <option value="graphite">Graphite</option>
          <option value="light-gray">Light gray</option>
          <option value="white">White</option>
        </select>
      </label>
      <button
        id="overviewBtn"
        className="hud-scene-only"
        disabled={!viewer}
        title="Frame the main reconstruction and cameras"
        onClick={() => viewer?.frameScene()}
      >
        Overview
      </button>
      <button
        id="prevCamBtn"
        className="hud-scene-only"
        disabled={!viewer || state.busy}
        title="Previous camera"
        onClick={() => viewer?.prevCam()}
      >
        <ChevronLeft size={14} />
      </button>
      <button
        id="nextCamBtn"
        className="hud-scene-only"
        disabled={!viewer || state.busy}
        title="Next camera"
        onClick={() => viewer?.nextCam()}
      >
        <ChevronRight size={14} />
      </button>
      <button id="toggleStats" disabled={!viewer} onClick={() => viewer?.toggleStatsVisible()}>
        {state.statsVisible ? "Hide stats" : "Show stats"}
      </button>
      <label className="hud-scene-only">
        <input
          type="checkbox"
          id="toggleCams"
          checked={state.showCams}
          disabled={!viewer}
          onChange={(event) => viewer?.setShowCams(event.target.checked)}
        />{" "}
        Cameras
      </label>
      <label className="hud-scene-only">
        Point size{" "}
        <input
          type="range"
          id="ptSize"
          min="1"
          max="5"
          value={state.pointSize}
          disabled={!viewer}
          onChange={(event) => viewer?.setPointSize(Number(event.target.value))}
        />
      </label>
      <button
        id="toggleGround"
        type="button"
        disabled={!viewer}
        aria-pressed={state.showGround}
        className={state.showGround ? "is-active" : ""}
        onClick={() => {
          viewer?.toggleGroundVisible();
          viewer?.savePreferences();
        }}
      >
        {state.showGround ? "Hide plane" : "Show plane"}
      </button>
      <label className="plane-height-control">
        Plane Z{" "}
        <input
          type="range"
          id="groundY"
          min="-5"
          max="5"
          step="0.1"
          value={state.groundY}
          disabled={!viewer}
          aria-label="Plane height along reconstruction Z"
          onChange={(event) => {
            viewer?.setGroundY(Number(event.target.value));
            viewer?.savePreferences();
          }}
        />
        <output id="groundYValue" htmlFor="groundY">
          {state.groundY.toFixed(1)}
        </output>
      </label>
    </div>
  );
}

export function ViewerLoading() {
  const {
    state: { loading },
  } = useViewer();
  return (
    <div
      id="loadingOverlay"
      className={`loading-overlay${loading.active ? " active" : ""}${loading.compact ? " compact" : ""}`}
      role="status"
    >
      <div className="loading-box">
        <span id="loadingMessage">{loading.message}</span>
        <div className="loading-progress-track">
          <div className="loading-progress-fill" id="loadingProgress" style={{ width: `${loading.progress * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

/** Keep frequent geometry statistics out of the run form and log component tree. */
export function SceneStats({ children }: { children?: ReactNode }) {
  const { state } = useViewer();
  return (
    <div
      id="sceneStats"
      style={{ display: state.statsVisible ? undefined : "none" }}
      className={children ? "dask-active" : undefined}
    >
      <div className="stat-group" data-mode="scene" style={{ display: state.mode === "splat" ? "none" : "grid" }}>
        <div className="stat-pair">
          <span className="label">Cameras</span>
          <span className="value" id="statCameras">
            {state.cameras.toLocaleString()}
          </span>
        </div>
        <div className="stat-pair">
          <span className="label">Points</span>
          <span className="value" id="statPoints">
            {state.points.toLocaleString()}
          </span>
        </div>
        <div className="stat-wide">
          <span className="label">Image</span>
          <span className="value" id="statImageName">
            {state.imageName || "—"}
          </span>
        </div>
      </div>
      <div className="stat-group" data-mode="splat" style={{ display: state.mode === "splat" ? "grid" : "none" }}>
        <div className="stat-pair">
          <span className="label">Splats</span>
          <span className="value" id="statSplats">
            {state.splats.toLocaleString()}
          </span>
        </div>
      </div>
      {children}
    </div>
  );
}
