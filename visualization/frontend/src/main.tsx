import React, {
  type Dispatch,
  type FormEvent,
  type InputHTMLAttributes,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal, flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import * as Collapsible from "@radix-ui/react-collapsible";
import * as Progress from "@radix-ui/react-progress";
import * as Switch from "@radix-ui/react-switch";
import * as Tabs from "@radix-ui/react-tabs";
import { Activity, Box, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, CircleAlert, Copy, Cpu, Database, Download, FolderUp, Github, Maximize2, Minus, MonitorCog, Play, RefreshCw, Server, SlidersHorizontal, Square, Terminal, X, ArrowRight, Layers3, PanelLeft, Plus, Images, Orbit } from "lucide-react";
import "./styles.css";
import "./workspace.css";
import { ExperimentDetails } from "./experiment-details";
import { PhotoPanel } from "./photo-panel";
import { useJobStream } from "./use-job-stream";
import { ResultsBrowser } from "./results-browser";
import { waitForViewerIdle, type ResultItem } from "./result-view";
import { ViewerCanvas, ViewerProvider, useViewerApi } from "./viewer-context";
import { ViewControls, ViewerToolbar, ViewerLoading, SceneStats, ViewerEmpty } from "./viewer-controls";

type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
type InputValue = string | number;
type FormValue = string | number | boolean;
type LoaderValues = Record<string, FormValue>;

interface Choice {
  id: string;
  label: string;
  status?: string;
  description?: string;
  live?: boolean;
}

interface ModelChoice extends Choice {
  capabilities: {
    iterative_splat: boolean;
    mvs: boolean;
    share_intrinsics: boolean;
  };
}

interface LoaderDescriptor {
  name: string;
  label: string;
  type: "boolean" | "integer" | "number" | "string";
  required: boolean;
  default: FormValue | null;
}

interface RunFormState {
  name: string;
  max_run_minutes: InputValue;
  sample_id: string;
  dataset_dir: string;
  images_dir: string;
  loader: string;
  config_name: string;
  splat_implementation: string;
  gaussian_splatting_config_name: string;
  gs_max_steps: InputValue;
  live_preview_interval: InputValue;
  run_mvs: boolean;
  execution_target: "local" | "remote";
  hardware: string;
  remote_endpoint: string;
  modal_api_key: string;
  modal_gpu: string;
  remote_hardware: string;
  max_resolution: InputValue;
  num_workers: InputValue;
  threads_per_worker: InputValue;
  worker_memory_limit: string;
  graph_partitioner: string;
  global_descriptor_config_name: string;
  retriever_config_name: string;
  correspondence_generator_config_name: string;
  verifier_config_name: string;
  max_frame_lookahead: InputValue;
  num_matched: InputValue;
  share_intrinsics: boolean;
  log: string;
  dashboard_port: string;
  input_worker: string;
  dask_tmpdir: string;
  dask_worker_hosts: string;
  dask_scheduler_host: string;
  dask_scheduler_port: InputValue;
  dask_worker_python: string;
  cluster_config: string;
  num_retry_cluster_connection: InputValue;
  advanced_overrides: string;
}

interface ConfigurationSchema {
  defaults: Partial<RunFormState> & Pick<RunFormState, "loader" | "config_name" | "splat_implementation">;
  models: ModelChoice[];
  loaders: string[];
  loader_options: Record<string, LoaderDescriptor[]>;
  graph_partitioners: string[];
  global_descriptors: string[];
  retrievers: string[];
  correspondence_generators: string[];
  verifiers: string[];
  log_levels: string[];
  gaussian_splatting_models: string[];
  splat_implementations: Choice[];
}

type HydraOverrideMap = Record<string, string>;

interface HydraFieldDescriptor {
  key: string;
  label: string;
  type: "boolean" | "integer" | "number" | "string";
  defaultValue: boolean | number | string;
  group: string;
  help: string;
  min?: number;
  max?: number;
  step?: number;
}

interface HydraParameterSchema {
  model: { preset: string; fields: HydraFieldDescriptor[] };
  gaussian: { preset: string; fields: HydraFieldDescriptor[] };
}

interface HardwareDevice extends Choice {
  kind: "cpu" | "cuda" | "nvidia" | "rocm" | "mps" | string;
  details: string;
  memory?: string;
  supports_gaussian_splatting: boolean;
}

interface HardwareCatalog {
  devices: HardwareDevice[];
  summary: string;
  platform?: {
    system?: string;
    machine?: string;
    torch?: string | null;
    cuda?: string | null;
  };
}

type SetupCheckState = "ready" | "warning" | "error" | "optional";

interface SetupCheck {
  id: string;
  label: string;
  state: SetupCheckState;
  detail: string;
  required: boolean;
  category: "runtime" | "system" | "submodules" | "hardware";
  action?: {
    label: string;
    enabled: boolean;
    reason?: string;
  };
}

interface SetupStatus {
  status: "ready" | "warning" | "error";
  summary: string;
  checked_at: string;
  counts: Record<SetupCheckState, number>;
  items: SetupCheck[];
}

interface RemoteWorkspace {
  configuration: ConfigurationSchema;
  hardware: HardwareCatalog;
  verified?: boolean;
}

interface ModalDiscovery {
  endpoint: string;
  app_name: string;
  function_name: string;
  api_key: string;
}

interface ModalDeployment {
  id: string;
  gpu: string;
  cpu?: number;
  memory_mb?: number;
  status: "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
  stage: string;
  phase?: "queued" | "building" | "deploying" | "verifying" | "ready" | "cancelled";
  log_tail: string[];
  endpoint: string;
  api_key: string;
  error: string;
}

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const temporary = document.createElement("textarea");
    temporary.value = text;
    temporary.style.position = "fixed";
    temporary.style.opacity = "0";
    document.body.appendChild(temporary);
    temporary.focus();
    temporary.select();
    try { return document.execCommand("copy"); }
    catch { return false; }
    finally { temporary.remove(); }
  }
}

function ModalDeploymentDialog({ deployment, onCancel, onClose }: { deployment: ModalDeployment; onCancel: () => void; onClose: () => void }) {
  const logRef = useRef<HTMLPreElement>(null);
  const copyReset = useRef<number | null>(null);
  const [copied, setCopied] = useState(false);
  const logs = deployment.log_tail.join("\n") || "Waiting for Modal build output…";

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [deployment.log_tail.length]);

  useEffect(() => () => { if (copyReset.current !== null) window.clearTimeout(copyReset.current); }, []);

  const copyLogs = async () => {
    if (!await copyTextToClipboard(logs)) return;
    setCopied(true);
    if (copyReset.current !== null) window.clearTimeout(copyReset.current);
    copyReset.current = window.setTimeout(() => setCopied(false), 1000);
  };

  return createPortal(<div className="modal-log-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="modal-log-dialog" role="dialog" aria-modal="true" aria-labelledby="modal-log-title">
      <header>
        <div><span>MODAL SETUP &amp; DEPLOYMENT</span><strong id="modal-log-title">{deployment.stage}</strong></div>
        <small>{deployment.gpu}{deployment.cpu ? ` · ${deployment.cpu} CPU` : ""}{deployment.memory_mb ? ` · ${formatBytes(deployment.memory_mb * 1024 * 1024)}` : ""} · {deployment.status}</small>
        {["queued", "running"].includes(deployment.status) && deployment.phase !== "verifying" && <button className="modal-dialog-stop" type="button" onClick={onCancel} title="Stop Modal setup"><Square size={11} fill="currentColor"/> Stop</button>}
        <button type="button" onClick={copyLogs} title="Copy deployment logs">{copied ? <Check size={13}/> : <Copy size={13}/>} {copied ? "Copied!" : "Copy"}</button>
        <button type="button" onClick={onClose} title="Close deployment logs" aria-label="Close deployment logs"><X size={15}/></button>
      </header>
      <pre ref={logRef}>{logs}</pre>
    </section>
  </div>, document.body);
}

const MODAL_WORKSPACE_STEPS = [
  { id: "building", label: "Prepare CUDA image", detail: "Install and cache the GTSFM environment" },
  { id: "deploying", label: "Deploy workspace", detail: "Publish the FastAPI workspace on Modal" },
  { id: "verifying", label: "Start & verify workspace", detail: "Check the CPU control service before launching a run" },
] as const;

function ModalDeploymentProgress({ deployment, onCancel, onExpand }: { deployment: ModalDeployment; onCancel: () => void; onExpand: () => void }) {
  const ready = deployment.phase === "ready";
  const verifying = deployment.phase === "verifying";
  const currentPhase = ready ? "verifying" : deployment.phase ?? "building";
  const currentIndex = Math.max(0, MODAL_WORKSPACE_STEPS.findIndex((step) => step.id === currentPhase));
  return <section className={`modal-deployment ${deployment.status}`} aria-label="Modal workspace progress">
    <div className="modal-deployment-heading"><span>{ready ? <Check size={12}/> : deployment.status === "failed" ? <CircleAlert size={12}/> : deployment.status === "cancelled" ? <Minus size={12}/> : <RefreshCw size={12}/>}<strong>{deployment.stage}</strong></span><span className="modal-deployment-actions"><small>{deployment.gpu}</small>{["queued", "running"].includes(deployment.status) && !verifying && <button className="modal-stop-action" type="button" onClick={onCancel} title="Stop Modal setup"><Square size={9} fill="currentColor"/> Stop</button>}<button type="button" onClick={onExpand} title="View all setup logs" aria-label="View all setup logs"><Maximize2 size={12}/></button></span></div>
    <ol className="modal-deployment-steps">
      {MODAL_WORKSPACE_STEPS.map((step, index) => {
        const complete = ready || index < currentIndex;
        const active = !ready && index === currentIndex;
        const failed = active && deployment.status === "failed";
        const cancelled = active && deployment.status === "cancelled";
        return <li key={step.id} data-state={failed ? "failed" : cancelled ? "cancelled" : complete ? "complete" : active ? "active" : "pending"}>
          <span className="modal-step-mark">{complete ? <Check size={10}/> : failed ? <X size={10}/> : cancelled ? <Minus size={10}/> : index + 1}</span>
          <div><strong>{step.label}</strong><small>{step.detail}</small></div>
        </li>;
      })}
    </ol>
    <div className="modal-log-preview-heading"><span>LIVE SETUP LOGS</span><button type="button" onClick={onExpand}><Maximize2 size={10}/> View all logs</button></div>
    <pre>{deployment.log_tail.slice(-6).join("\n") || "Waiting for Modal setup output…"}</pre>
  </section>;
}

interface JobEvent {
  job: Job;
  live: LiveState;
}

interface Job {
  history_note?: string | null;
  retry_available?: boolean;
  created_at?: string;
  id: string;
  name: string;
  status: JobStatus;
  command?: string[];
  output_root?: string;
  return_code?: number | null;
  spec: {
    config_name?: string;
    splat_implementation?: string;
    remote_connection?: string;
  };
  error?: string;
  remote?: { workspace_url?: string; cancel_status?: string };
  log_tail?: string[];
  has_final_splat?: boolean;
}

interface JobsResponse {
  items: Job[];
}

interface LiveState {
  progress?: number;
  stage?: string;
  message?: string;
  step?: number;
  max_steps?: number;
  loss?: number;
  splat_count?: number;
  preview_url?: string;
  preview_version?: string | number;
  final_url?: string;
  dask?: {
    workers: number;
    threads: number;
    running_tasks: number;
    completed_tasks: number;
    pending_tasks: number;
    failed_tasks: number;
    memory_bytes: number;
    memory_limit_bytes: number;
    cpu_percent: number;
    dashboard_url?: string;
  };
}

interface UploadedFolder {
  id: string;
  name: string;
  path: string;
  file_count: number;
  bytes: number;
  analysis: DatasetAnalysis;
  uploaded_at?: number;
}

interface DatasetAnalysis {
  image_count: number;
  image_bytes: number;
  total_megapixels: number;
  average_megapixels: number;
  max_width: number;
  max_height: number;
}

interface SampleDataset extends Choice {
  description: string;
  image_count: number;
  source_url: string;
  prepared: boolean;
  recommendations: {
    loader: string;
    config_name: string;
    max_resolution?: number;
    loader_options?: LoaderValues;
  };
}

interface SamplesResponse {
  items: SampleDataset[];
}

interface PreparedSample {
  path: string;
  sample: SampleDataset;
  analysis: DatasetAnalysis;
}

interface FolderFile {
  file: File;
  relativePath: string;
}

const EMPTY_FORM: RunFormState = {
  max_run_minutes: 60,
  name: "my-scene", sample_id: "", dataset_dir: "", images_dir: "", loader: "olsson", config_name: "vggt",
  splat_implementation: "gsplat", gaussian_splatting_config_name: "base_gs", gs_max_steps: 7000,
  live_preview_interval: 250, run_mvs: false, execution_target: "local", hardware: "cpu",
  remote_endpoint: "", modal_api_key: "", modal_gpu: "L40S", remote_hardware: "cpu", max_resolution: "", num_workers: 1,
  threads_per_worker: 1, worker_memory_limit: "32GB", graph_partitioner: "", global_descriptor_config_name: "",
  retriever_config_name: "", correspondence_generator_config_name: "", verifier_config_name: "",
  max_frame_lookahead: "", num_matched: "", share_intrinsics: false, log: "INFO", dashboard_port: "",
  input_worker: "", dask_tmpdir: "", dask_worker_hosts: "", dask_scheduler_host: "", dask_scheduler_port: 0, dask_worker_python: "", cluster_config: "", num_retry_cluster_connection: "", advanced_overrides: "",
};

// Keep the form useful on the very first paint. The full catalog is loaded from
// the workspace in the background and replaces this small, stable core catalog.
const BOOTSTRAP_SCHEMA: ConfigurationSchema = {
  defaults: {
    config_name: "vggt",
    loader: "olsson",
    splat_implementation: "gsplat",
    num_workers: 1,
    threads_per_worker: 1,
    worker_memory_limit: "32GB",
    run_mvs: false,
    gs_max_steps: 7000,
    live_preview_interval: 250,
    hardware: "cpu",
  },
  models: [{
    id: "vggt",
    label: "VGGT",
    capabilities: { iterative_splat: true, mvs: false, share_intrinsics: false },
  }],
  loaders: ["olsson"],
  loader_options: { olsson: [] },
  graph_partitioners: [],
  global_descriptors: [],
  retrievers: [],
  correspondence_generators: [],
  verifiers: [],
  log_levels: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
  gaussian_splatting_models: ["base_gs"],
  splat_implementations: [
    { id: "none", label: "No splats", description: "Run reconstruction only.", live: false },
    { id: "gsplat", label: "Optimized Gaussian splats", description: "Iteratively optimize gsplat Gaussians and show live previews.", live: true },
    { id: "anysplat", label: "AnySplat", description: "Generate splats with the feed-forward AnySplat model.", live: false },
  ],
};

function HydraFieldControl({ descriptor, value, onChange }: { descriptor: HydraFieldDescriptor; value?: string; onChange: (value: string) => void }) {
  const inherited = value === undefined || value === "";
  return <div className={`hydra-field ${inherited ? "inherited" : "overridden"}`}>
    <div className="hydra-field-heading"><span>{descriptor.label}</span>{!inherited && <button type="button" onClick={() => onChange("")} title={`Reset ${descriptor.label}`}>Reset</button>}</div>
    {descriptor.type === "boolean"
      ? <select value={value ?? ""} onChange={(event) => onChange(event.target.value)} aria-label={descriptor.label}>
        <option value="">Preset default · {descriptor.defaultValue ? "On" : "Off"}</option>
        <option value="true">On</option><option value="false">Off</option>
      </select>
      : <input type={descriptor.type === "string" ? "text" : "number"} value={value ?? ""} min={descriptor.min} max={descriptor.max} step={descriptor.type === "string" ? undefined : descriptor.step ?? (descriptor.type === "integer" ? 1 : "any")} placeholder={`Preset default · ${descriptor.defaultValue}`} onChange={(event) => onChange(event.target.value)} aria-label={descriptor.label} />}
    <small>{descriptor.help}</small>
    <code>{descriptor.key}</code>
  </div>;
}

function HydraFieldGroups({ fields, values, onChange }: { fields: HydraFieldDescriptor[]; values: HydraOverrideMap; onChange: (key: string, value: string) => void }) {
  const groups = fields.reduce<Record<string, HydraFieldDescriptor[]>>((result, field) => {
    (result[field.group] ??= []).push(field);
    return result;
  }, {});
  return <>{Object.entries(groups).map(([title, groupFields]) => <HydraFieldGroup key={title} title={title} fields={groupFields} values={values} onChange={onChange} />)}</>;
}

interface ParsedJsonConfiguration {
  model: HydraOverrideMap;
  gaussian: HydraOverrideMap;
  expert: string[];
}

const jsonValueForOverride = (field: HydraFieldDescriptor, value?: string): boolean | number | string | null => {
  if (value === undefined || value === "") return null;
  if (field.type === "boolean") return value === "true";
  if (field.type === "integer" || field.type === "number") return Number(value);
  return value;
};

function configurationJsonTemplate(parameters: HydraParameterSchema, model: HydraOverrideMap, gaussian: HydraOverrideMap, expert: string): string {
  return JSON.stringify({
    model: Object.fromEntries(parameters.model.fields.map((field) => [field.key, jsonValueForOverride(field, model[field.key])])),
    gaussian: Object.fromEntries(parameters.gaussian.fields.map((field) => [field.key, jsonValueForOverride(field, gaussian[field.key])])),
    expert: expert.split("\n").map((line) => line.trim()).filter(Boolean),
  }, null, 2);
}

function flattenJsonConfiguration(value: unknown, prefix = "", output: Record<string, unknown> = {}): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error((prefix || "Configuration") + " must be a JSON object.");
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? prefix + "." + key : key;
    if (child !== null && typeof child === "object" && !Array.isArray(child)) flattenJsonConfiguration(child, path, output);
    else output[path] = child;
  }
  return output;
}

function parseJsonConfiguration(value: string, parameters: HydraParameterSchema): ParsedJsonConfiguration {
  const parsed: unknown = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("The pasted configuration must be a JSON object.");
  const root = parsed as Record<string, unknown>;
  const hasSections = ["model", "gaussian", "expert", "hydra_overrides", "gaussian_splatting_overrides", "advanced_overrides"].some((key) => key in root);
  const modelSource = root.model ?? root.hydra_overrides ?? (hasSections ? {} : root);
  const gaussianSource = root.gaussian ?? root.gaussian_splatting_overrides ?? {};
  const expertSource = root.expert ?? root.advanced_overrides ?? [];
  const readSection = (source: unknown, fields: HydraFieldDescriptor[], section: string): HydraOverrideMap => {
    const descriptors = new Map(fields.map((field) => [field.key, field]));
    const result: HydraOverrideMap = {};
    const unknown: string[] = [];
    for (const [rawKey, rawValue] of Object.entries(flattenJsonConfiguration(source))) {
      const key = section === "Gaussian" ? rawKey.replace(/^gaussian_splatting_optimizer\.cfg\./, "") : rawKey;
      const descriptor = descriptors.get(key);
      if (!descriptor) { unknown.push(rawKey); continue; }
      if (rawValue === null) continue;
      const valid = descriptor.type === "boolean"
        ? typeof rawValue === "boolean"
        : descriptor.type === "integer"
          ? typeof rawValue === "number" && Number.isInteger(rawValue)
          : descriptor.type === "number"
            ? typeof rawValue === "number" && Number.isFinite(rawValue)
            : typeof rawValue === "string";
      if (!valid) throw new Error(section + " setting “" + rawKey + "” must be a " + descriptor.type + " value.");
      result[key] = String(rawValue);
    }
    if (unknown.length) throw new Error("Unknown " + section.toLowerCase() + " setting" + (unknown.length === 1 ? "" : "s") + ": " + unknown.slice(0, 4).join(", ") + (unknown.length > 4 ? "…" : ""));
    return result;
  };
  if (!Array.isArray(expertSource) || !expertSource.every((entry) => typeof entry === "string")) {
    throw new Error("Expert overrides must be a JSON array of Hydra key=value strings.");
  }
  const invalidExpert = expertSource.find((entry) => !entry.includes("=") || entry.startsWith("--") || entry.includes("\0"));
  if (invalidExpert) throw new Error("Expert override “" + invalidExpert + "” must use Hydra key=value syntax.");
  return {
    model: readSection(modelSource, parameters.model.fields, "Model"),
    gaussian: readSection(gaussianSource, parameters.gaussian.fields, "Gaussian"),
    expert: expertSource.map((entry) => entry.trim()).filter(Boolean),
  };
}

function ConfigurationJsonDialog({ parameters, model, gaussian, expert, onApply, onClose }: {
  parameters: HydraParameterSchema;
  model: HydraOverrideMap;
  gaussian: HydraOverrideMap;
  expert: string;
  onApply: (configuration: ParsedJsonConfiguration) => void;
  onClose: () => void;
}) {
  const template = useMemo(() => configurationJsonTemplate(parameters, model, gaussian, expert), [parameters, model, gaussian, expert]);
  const [value, setValue] = useState(template);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const copyReset = useRef<number | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);
  useEffect(() => () => { if (copyReset.current !== null) window.clearTimeout(copyReset.current); }, []);

  const copyTemplate = async () => {
    if (!await copyTextToClipboard(value)) return;
    setCopied(true);
    if (copyReset.current !== null) window.clearTimeout(copyReset.current);
    copyReset.current = window.setTimeout(() => setCopied(false), 1000);
  };
  const apply = () => {
    try {
      onApply(parseJsonConfiguration(value, parameters));
      onClose();
    } catch (reason) {
      setError(reason instanceof SyntaxError ? "Invalid JSON: " + reason.message : errorMessage(reason));
    }
  };

  return createPortal(<div className="configuration-json-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="configuration-json-dialog" role="dialog" aria-modal="true" aria-labelledby="configuration-json-title">
      <header><div><span>HYDRA CONFIGURATION</span><strong id="configuration-json-title">Paste JSON configuration</strong></div><button type="button" onClick={onClose} aria-label="Close JSON configuration"><X size={15}/></button></header>
      <p>The template lists every safe setting for the selected presets. Leave a value as <code>null</code> to inherit the preset default. Nested objects and flat dotted Hydra keys are both accepted.</p>
      <textarea value={value} onChange={(event) => { setValue(event.target.value); setError(""); }} spellCheck={false} aria-label="JSON configuration" />
      {error && <div className="configuration-json-error"><CircleAlert size={13}/><span>{error}</span></div>}
      <footer><button type="button" onClick={() => { setValue(template); setError(""); }}>Reset template</button><span/><button type="button" onClick={copyTemplate}>{copied ? <Check size={12}/> : <Copy size={12}/>} {copied ? "Copied!" : "Copy"}</button><button className="primary" type="button" onClick={apply}>Apply configuration</button></footer>
    </section>
  </div>, document.body);
}

function HydraFieldGroup({ title, fields, values, onChange }: { title: string; fields: HydraFieldDescriptor[]; values: HydraOverrideMap; onChange: (key: string, value: string) => void }) {
  return <section className="hydra-group"><header><strong>{title}</strong><small>{fields.filter((field) => values[field.key]).length} changed</small></header><div className="hydra-grid">{fields.map((field) => <HydraFieldControl key={field.key} descriptor={field} value={values[field.key]} onChange={(value) => onChange(field.key, value)} />)}</div></section>;
}

const displayName = (value: unknown): string => String(value ?? "").split("_").map((token) => {
  if (["api", "ba", "colmap", "gpu", "gs", "mvs", "sift", "vggt"].includes(token)) return token.toUpperCase();
  if (token === "anysplat") return "AnySplat";
  if (token === "megaloc") return "MegaLoc";
  return token.charAt(0).toUpperCase() + token.slice(1);
}).join(" ");

const errorMessage = (reason: unknown): string => reason instanceof Error ? reason.message : String(reason);

async function getJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || payload.detail || `Request failed (${response.status})`);
  return payload as T;
}

function Brand({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return <div className="studio-identity">
    <img src="/static/brand/sfm-logo.png" alt="GTSfM" className="gtsfm-wordmark" />
    <span className="studio-name">Studio</span>
    <button className="library-toggle" type="button" onClick={onToggle} title={collapsed ? "Expand side panel" : "Collapse side panel"} aria-label={collapsed ? "Expand side panel" : "Collapse side panel"}><PanelLeft size={17}/></button>
  </div>;
}

interface FieldProps {
  label: string;
  optional?: boolean;
  children: ReactNode;
}

function Field({ label, optional = false, children }: FieldProps) {
  return <label>{label}{optional && <span className="optional">optional</span>}{children}</label>;
}

interface SelectFieldProps {
  label: string;
  value: InputValue | null | undefined;
  options: Array<string | Choice>;
  onChange: (value: string) => void;
  disabled?: boolean;
  empty?: string | null;
  id?: string;
}

function SelectField({ label, value, options, onChange, disabled = false, empty = null, id }: SelectFieldProps) {
  const selected = value ?? "";
  const available = (selected === "" && empty !== null) || options.some((option) => (typeof option === "string" ? option : option.id) === String(selected));
  return <Field label={label}><select id={id} aria-label={label} aria-invalid={!available} value={selected} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
    {!available && <option value={selected} disabled>{selected ? `${selected} (unavailable)` : "Choose an option…"}</option>}
    {empty !== null && <option value="">{empty}</option>}
    {options.map((option) => {
      const item = typeof option === "string" ? { id: option, label: displayName(option) } : option;
      return <option key={item.id} value={item.id} disabled={Boolean(item.status && item.status !== "available")}>{item.label}</option>;
    })}
  </select></Field>;
}

interface TextFieldProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange"> {
  label: string;
  optional?: boolean;
  value: InputValue | null | undefined;
  onChange: (value: string) => void;
}

function TextField({ label, optional = false, value, onChange, ...props }: TextFieldProps) {
  return <Field label={label} optional={optional}><input value={value ?? ""} onChange={(event) => onChange(event.target.value)} {...props} /></Field>;
}

const formatBytes = (value: number): string => {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount < 10 && unit > 0 ? amount.toFixed(1) : Math.round(amount)} ${units[unit]}`;
};

interface ModalGpuChoice extends Choice {
  pricePerSecond: number;
  relativeSpeed: number;
  memoryGiB: number;
}

const MODAL_GPU_CHOICES: ModalGpuChoice[] = [
  { id: "T4", label: "NVIDIA T4 · 16 GB", pricePerSecond: 0.000164, relativeSpeed: 0.3, memoryGiB: 16 },
  { id: "L4", label: "NVIDIA L4 · 24 GB", pricePerSecond: 0.000222, relativeSpeed: 0.5, memoryGiB: 24 },
  { id: "A10", label: "NVIDIA A10 · 24 GB", pricePerSecond: 0.000306, relativeSpeed: 0.58, memoryGiB: 24 },
  { id: "L40S", label: "NVIDIA L40S · 48 GB", pricePerSecond: 0.000542, relativeSpeed: 1, memoryGiB: 48 },
  { id: "A100-40GB", label: "NVIDIA A100 · 40 GB", pricePerSecond: 0.000583, relativeSpeed: 1.18, memoryGiB: 40 },
  { id: "A100-80GB", label: "NVIDIA A100 · 80 GB", pricePerSecond: 0.000694, relativeSpeed: 1.3, memoryGiB: 80 },
  { id: "RTX-PRO-6000", label: "NVIDIA RTX PRO 6000 · 96 GB", pricePerSecond: 0.000842, relativeSpeed: 1.45, memoryGiB: 96 },
  { id: "H100", label: "NVIDIA H100 · 80 GB", pricePerSecond: 0.001097, relativeSpeed: 1.8, memoryGiB: 80 },
  { id: "H200", label: "NVIDIA H200 · 141 GB", pricePerSecond: 0.001261, relativeSpeed: 1.95, memoryGiB: 141 },
  { id: "B200", label: "NVIDIA B200 · 180 GB", pricePerSecond: 0.001736, relativeSpeed: 2.35, memoryGiB: 180 },
  { id: "B300", label: "NVIDIA B300 · 288 GB · Coming Soon!", status: "coming-soon", pricePerSecond: 0.001972, relativeSpeed: 2.65, memoryGiB: 288 },
];

interface ModalGpuRecommendation {
  gpu: ModalGpuChoice;
  imageCount: number;
  effectiveMegapixels: number;
  estimatedMemoryGiB: number;
  reason: string;
}

function recommendModalGpu(form: RunFormState, analysis?: DatasetAnalysis, fallbackImageCount = 0): ModalGpuRecommendation | null {
  const imageCount = analysis?.image_count || fallbackImageCount;
  if (!imageCount) return null;

  const sourceMegapixels = analysis?.average_megapixels && analysis.average_megapixels > 0 ? analysis.average_megapixels : 8;
  const maxResolution = Number(form.max_resolution);
  const cappedMegapixels = Number.isFinite(maxResolution) && maxResolution > 0 ? (maxResolution ** 2 * 0.75) / 1_000_000 : sourceMegapixels;
  const effectiveMegapixels = Math.max(0.25, Math.min(sourceMegapixels, cappedMegapixels));
  const sceneUnits = imageCount * Math.sqrt(effectiveMegapixels / 8);
  const modelMemory = form.config_name === "vggt" ? 5 : form.config_name.includes("fast") ? 3 : 7;
  const splatMemory = form.splat_implementation === "gsplat" ? 3 : form.splat_implementation === "anysplat" ? 6 : 0;
  const estimatedMemoryGiB = Math.ceil(5 + sceneUnits * 0.1 + modelMemory + splatMemory + (form.run_mvs ? 4 : 0));

  let gpuId = "L4";
  if (estimatedMemoryGiB <= 22) gpuId = "L4";
  else if (estimatedMemoryGiB <= 44) gpuId = "L40S";
  else if (estimatedMemoryGiB <= 72 && sceneUnits < 350) gpuId = "A100-80GB";
  else if (estimatedMemoryGiB <= 72) gpuId = "H100";
  else if (estimatedMemoryGiB <= 125) gpuId = "H200";
  else gpuId = "B200";

  const gpu = MODAL_GPU_CHOICES.find((item) => item.id === gpuId) ?? MODAL_GPU_CHOICES[3];
  const scale = imageCount < 40 ? "small" : imageCount < 140 ? "medium" : imageCount < 350 ? "large" : "very large";
  return {
    gpu,
    imageCount,
    effectiveMegapixels,
    estimatedMemoryGiB,
    reason: `${scale} ${imageCount}-image workload at about ${effectiveMegapixels.toFixed(effectiveMegapixels < 10 ? 1 : 0)} MP per image`,
  };
}

function memoryGiB(value: string): number {
  const match = value.trim().match(/^([\d.]+)\s*(gb|gib|mb|mib)?$/i);
  if (!match) return 32;
  const amount = Number(match[1]);
  return /m/i.test(match[2] || "") ? amount / 1024 : amount;
}

interface AdvancedMachineProfile {
  key: string;
  label: string;
  description: string;
  workers: number;
  threadsPerWorker: number;
  memoryPerWorkerGiB: number;
  allowWorkers: boolean;
  allowThreads: boolean;
  allowMemory: boolean;
  allowLocalRuntime: boolean;
}

const MACHINE_THREADS_KEY = "gtsfm-studio-machine-threads:";

function storedMachineThreads(profile: AdvancedMachineProfile): number {
  try {
    const saved = Number(window.localStorage.getItem(MACHINE_THREADS_KEY + profile.key));
    if (profile.allowThreads && Number.isSafeInteger(saved) && saved > 0) return saved;
  } catch { /* storage is optional */ }
  return profile.threadsPerWorker;
}

function rememberMachineThreads(profile: AdvancedMachineProfile, value: string): void {
  const threads = Number(value);
  if (!profile.allowThreads || !Number.isSafeInteger(threads) || threads < 1) return;
  try { window.localStorage.setItem(MACHINE_THREADS_KEY + profile.key, String(threads)); }
  catch { /* storage is optional */ }
}

const MODAL_VM_RESOURCES: Record<string, { cpu: number; memoryGiB: number }> = {
  T4: { cpu: 4, memoryGiB: 32 },
  L4: { cpu: 4, memoryGiB: 32 },
  A10: { cpu: 6, memoryGiB: 48 },
  L40S: { cpu: 8, memoryGiB: 64 },
  "A100-40GB": { cpu: 8, memoryGiB: 64 },
  "A100-80GB": { cpu: 12, memoryGiB: 96 },
  "RTX-PRO-6000": { cpu: 16, memoryGiB: 128 },
  H100: { cpu: 16, memoryGiB: 128 },
  H200: { cpu: 20, memoryGiB: 192 },
  B200: { cpu: 24, memoryGiB: 256 },
};

function advancedMachineProfile(form: RunFormState, hardware: HardwareCatalog | null, selected?: HardwareDevice): AdvancedMachineProfile {
  if (form.execution_target === "local" && form.dask_worker_hosts.trim()) {
    return {
      key: "ssh-workers", label: "SSH workers",
      description: "Worker settings apply on each SSH host. The control machine can use a different device.",
      workers: Number(form.num_workers), threadsPerWorker: Number(form.threads_per_worker),
      memoryPerWorkerGiB: memoryGiB(form.worker_memory_limit),
      allowWorkers: true, allowThreads: true, allowMemory: true, allowLocalRuntime: true,
    };
  }
  if (form.execution_target === "remote") {
      const resources = MODAL_VM_RESOURCES[form.modal_gpu] ?? MODAL_VM_RESOURCES.L40S;
      const gpu = MODAL_GPU_CHOICES.find((item) => item.id === form.modal_gpu);
      return {
        key: `modal:${form.modal_gpu}`,
        label: `Modal ${form.modal_gpu}`,
        description: `${gpu?.memoryGiB ?? 48} GB GPU · CPU and RAM are provisioned when this workspace is deployed. One worker is fixed to the single GPU.`,
        workers: 1,
        threadsPerWorker: resources.cpu,
        memoryPerWorkerGiB: resources.memoryGiB,
        allowWorkers: false,
        allowThreads: true,
        allowMemory: true,
        allowLocalRuntime: false,
      };
  }

  const cpu = hardware?.devices.find((device) => device.kind === "cpu");
  const logicalCores = Math.max(1, Number(cpu?.details.match(/(\d+)\s+logical cores/i)?.[1]) || 1);
  const hostMemoryGiB = Math.max(4, memoryGiB(cpu?.memory || "32GB"));
  const accelerator = Boolean(selected && selected.kind !== "cpu");
  const workers = accelerator ? 1 : Math.min(4, Math.max(1, Math.floor(logicalCores / 4)));
  const threadsPerWorker = accelerator
    ? Math.min(8, logicalCores)
    : Math.max(1, Math.floor(logicalCores / workers));
  const memoryPerWorkerGiB = Math.max(4, Math.floor((hostMemoryGiB * 0.75) / workers));
  return {
    key: `local:${selected?.id ?? "detecting"}:${logicalCores}:${Math.round(hostMemoryGiB)}`,
    label: selected?.label ?? "Detecting this machine",
    description: accelerator
      ? "One worker is assigned to the selected accelerator. CPU threads and host-memory limits remain adjustable."
      : "Worker count, CPU threads, and memory are tuned from the detected local resources.",
    workers,
    threadsPerWorker,
    memoryPerWorkerGiB,
    allowWorkers: Boolean(selected),
    allowThreads: Boolean(selected),
    allowMemory: Boolean(selected),
    allowLocalRuntime: true,
  };
}

async function collectEntry(entry: FileSystemEntry, parent = ""): Promise<FolderFile[]> {
  const path = parent ? `${parent}/${entry.name}` : entry.name;
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
    return [{ file, relativePath: path }];
  }
  if (!entry.isDirectory) return [];
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const children: FileSystemEntry[] = [];
  while (true) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    children.push(...batch);
  }
  return (await Promise.all(children.map((child) => collectEntry(child, path)))).flat();
}

async function filesFromDrop(dataTransfer: DataTransfer): Promise<FolderFile[]> {
  const entries = Array.from(dataTransfer.items)
    .map((item) => (item as DataTransferItem & { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry?.())
    .filter((entry): entry is FileSystemEntry => Boolean(entry));
  if (entries.length) return (await Promise.all(entries.map((entry) => collectEntry(entry)))).flat();
  return Array.from(dataTransfer.files).map((file) => ({ file, relativePath: file.name }));
}

interface FolderDropFieldProps {
  label: string;
  optional?: boolean;
  value: UploadedFolder | null;
  onUploaded: (folder: UploadedFolder | null) => void;
  onError: (message: string) => void;
}

function FolderDropField({ label, optional = false, value, onUploaded, onError }: FolderDropFieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const upload = async (items: FolderFile[]) => {
    if (!items.length) { onError("Choose a folder containing at least one file."); return; }
    setUploading(true); onError("");
    try {
      const data = new FormData();
      data.append("manifest", JSON.stringify(items.map((item) => item.relativePath)));
      items.forEach((item) => data.append("files", item.file, item.file.name));
      const folder = await getJson<UploadedFolder>("/api/uploads", { method: "POST", body: data });
      onUploaded(folder);
    } catch (reason) { onError(errorMessage(reason)); } finally { setUploading(false); }
  };

  return <div className="folder-field">
    <div className="folder-label">{label}{optional && <span className="optional">optional</span>}</div>
    <button type="button" className={`folder-drop ${dragging ? "dragging" : ""} ${value ? "has-folder" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }}
      onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
      onDrop={async (event) => { event.preventDefault(); setDragging(false); await upload(await filesFromDrop(event.dataTransfer)); }}
      disabled={uploading}>
      <span className="folder-icon">{value ? <Check size={17} /> : <FolderUp size={18} />}</span>
      <span className="folder-copy">{uploading ? <><strong>Importing folder…</strong><small>Keeping the directory structure intact</small></> : value ?
        <><strong>{value.name}</strong><small>{value.file_count} file{value.file_count === 1 ? "" : "s"} · {formatBytes(value.bytes)} · Click to replace</small></> :
        <><strong>Drop a folder here</strong><small>or click to choose one</small></>}</span>
    </button>
    <input ref={inputRef} className="folder-input" type="file" multiple
      {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
      onChange={async (event) => {
        const files = Array.from(event.target.files ?? []);
        await upload(files.map((file) => ({ file, relativePath: file.webkitRelativePath || file.name })));
        event.target.value = "";
      }} />
    {value && value.analysis.image_count > 0 && <div className="capture-preview" aria-label="Selected capture photographs">{Array.from({length: Math.min(3,value.analysis.image_count)}, (_,index) => <img key={index} src={`/api/uploads/${value.id}/preview?index=${index}`} alt={`${value.name} source photograph ${index+1}`} loading="lazy"/>)}</div>}
    {value && <button type="button" className="folder-remove" onClick={() => onUploaded(null)}>Remove selection</button>}
  </div>;
}

interface ToggleProps {
  id: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  children: ReactNode;
}

function Toggle({ id, checked, onCheckedChange, disabled = false, children }: ToggleProps) {
  return <div className={`toggle-row ${disabled ? "disabled" : ""}`}>
    <Switch.Root id={id} className="switch-root" checked={checked} onCheckedChange={onCheckedChange} disabled={disabled}>
      <Switch.Thumb className="switch-thumb" />
    </Switch.Root>
    <label htmlFor={id}>{children}</label>
  </div>;
}

interface SectionProps {
  number: string;
  title: string;
  subtitle: string;
  children: ReactNode;
  active?: boolean;
}

function Section({ number, title, subtitle, children, active = true }: SectionProps) {
  return <section className="form-section" id={`run-step-${number}`} hidden={!active} aria-labelledby={`step-title-${number}`}>
    <div className="section-heading"><span className="step-number">{number}</span><div><h2 id={`step-title-${number}`} tabIndex={-1}>{title}</h2><small>{subtitle}</small></div></div>
    {children}
  </section>;
}

interface LoaderOptionsProps {
  descriptors: LoaderDescriptor[];
  values: LoaderValues;
  setValues: Dispatch<SetStateAction<LoaderValues>>;
}

function LoaderOptions({ descriptors, values, setValues }: LoaderOptionsProps) {
  if (!descriptors.length) return null;
  return <div className="loader-options"><p className="mini-heading">Format-specific options</p>
    {descriptors.map((descriptor) => descriptor.type === "boolean" ?
      <SelectField key={descriptor.name} label={descriptor.label} value={String(values[descriptor.name] ?? descriptor.default ?? false)}
        options={[{ id: "true", label: "True" }, { id: "false", label: "False" }]}
        onChange={(value) => setValues((current) => ({ ...current, [descriptor.name]: value === "true" }))} /> :
      <TextField key={descriptor.name} label={descriptor.label} optional={!descriptor.required}
        required={descriptor.required} type={["integer", "number"].includes(descriptor.type) ? "number" : "text"}
        step={descriptor.type === "number" ? "any" : undefined} value={(values[descriptor.name] ?? descriptor.default ?? "") as InputValue}
        onChange={(value) => setValues((current) => ({ ...current, [descriptor.name]: value }))} />)}
  </div>;
}

function HardwareCard({ device }: { device?: HardwareDevice }) {
  if (!device) return <div className="hardware-card"><strong>Detecting hardware…</strong></div>;
  return <div className="hardware-card">
    <strong>Device capabilities</strong><small>{device.details}{device.memory ? ` · ${device.memory}` : ""}</small>
    <span className={`capability ${device.supports_gaussian_splatting ? "yes" : "no"}`}>{device.supports_gaussian_splatting ? "Splat ready" : "Reconstruction"}</span>
  </div>;
}

const EXECUTION_KEYS = [
  "execution_target", "hardware", "remote_endpoint", "remote_hardware", "modal_gpu",
  "num_workers", "threads_per_worker", "worker_memory_limit",
  "dashboard_port", "input_worker", "dask_tmpdir", "num_retry_cluster_connection",
  "dask_worker_hosts", "dask_scheduler_host", "dask_scheduler_port", "dask_worker_python",
] as const;
type ExecutionConfiguration = { name: string; settings: Partial<Pick<RunFormState, typeof EXECUTION_KEYS[number]>> };

interface RunFormProps {
  onPreparing: (preparing: boolean) => void;
  reuse?: { settings: Record<string, unknown>; version: number };
  schema: ConfigurationSchema;
  hardware: HardwareCatalog | null;
  samples: SampleDataset[];
  samplesLoading: boolean;
  onStarted: (job: Job) => void;
  schemaLoading: boolean;
  schemaError: string;
  onRetrySchema: () => void;
}

function RunForm({ onPreparing, reuse, schema, hardware, samples, samplesLoading, onStarted, schemaLoading, schemaError, onRetrySchema }: RunFormProps) {
  const [form, setForm] = useState<RunFormState>(EMPTY_FORM);
  const currentForm = useRef(form);
  currentForm.current = form;
  const [executionConfigurations, setExecutionConfigurations] = useState<ExecutionConfiguration[]>([]);
  const [executionName, setExecutionName] = useState("");
  const [selectedExecution, setSelectedExecution] = useState("");
  const [executionMessage, setExecutionMessage] = useState("");
  const [executionBusy, setExecutionBusy] = useState(false);
  const loadedExecution = useRef(false);
  const [inputMode, setInputMode] = useState<"upload" | "sample" | "previous">("upload");
  const [previousUploads, setPreviousUploads] = useState<UploadedFolder[]>([]);
  const [uploadsLoading, setUploadsLoading] = useState(false);
  const [uploadsError, setUploadsError] = useState("");
  const [previousFolder, setPreviousFolder] = useState<UploadedFolder | null>(null);
  const [previousImages, setPreviousImages] = useState<UploadedFolder | null>(null);
  const refreshUploads = async () => {
    setUploadsLoading(true); setUploadsError("");
    try { setPreviousUploads((await getJson<{ items: UploadedFolder[] }>("/api/uploads")).items); }
    catch (reason) { setUploadsError(errorMessage(reason)); }
    finally { setUploadsLoading(false); }
  };
  const [loaderOptions, setLoaderOptions] = useState<LoaderValues>({});
  const [remote, setRemote] = useState<RemoteWorkspace | null>(null);
  const [remoteMessage, setRemoteMessage] = useState("");
  const [account, setAccount] = useState<{status: string; profile: string; workspace: string; profiles: string[]; login_url: string; code: string; error: string} | null>(null);
  const [accountBusy, setAccountBusy] = useState(false);
  const [reuseMessage, setReuseMessage] = useState("");
  const connectAccount = async (signIn = false, profile = account?.profile || "") => {
    setAccountBusy(true); setError("");
    try { setAccount(await getJson("/api/modal/connect", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({profile, sign_in: signIn})})); }
    catch (reason) { setError(errorMessage(reason)); }
    finally { setAccountBusy(false); }
  };
  useEffect(() => {
    if (form.execution_target !== "remote") return;
    void getJson<typeof account>("/api/modal/account").then(setAccount).catch(() => {});
  }, [form.execution_target]);
  useEffect(() => {
    if (account?.status !== "signing_in") return;
    const timer = window.setInterval(() => { void getJson<typeof account>("/api/modal/account").then(setAccount).catch(() => {}); }, 1000);
    return () => window.clearInterval(timer);
  }, [account?.status]);
  const [remoteChecking, setRemoteChecking] = useState(false);
  const [modalDeploying, setModalDeploying] = useState(false);
  const [modalWorkspaceIssue, setModalWorkspaceIssue] = useState(false);
  const [modalDeployment, setModalDeployment] = useState<ModalDeployment | null>(null);
  const [modalLogExpanded, setModalLogExpanded] = useState(false);
  const [configurationAdvanced, setConfigurationAdvanced] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [modelHydraOverrides, setModelHydraOverrides] = useState<HydraOverrideMap>({});
  const [splatHydraOverrides, setSplatHydraOverrides] = useState<HydraOverrideMap>({});
  const [hydraParameters, setHydraParameters] = useState<HydraParameterSchema | null>(null);
  const [hydraParametersLoading, setHydraParametersLoading] = useState(true);
  const [hydraParametersError, setHydraParametersError] = useState("");
  const [hydraParametersRevision, setHydraParametersRevision] = useState(0);
  const [jsonConfigurationOpen, setJsonConfigurationOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const submissionPending = useRef(false);
  useEffect(() => { onPreparing(busy); }, [busy, onPreparing]);
  const [error, setError] = useState("");
  const [datasetFolder, setDatasetFolder] = useState<UploadedFolder | null>(null);
  const [imagesFolder, setImagesFolder] = useState<UploadedFolder | null>(null);
  const [preparedSample, setPreparedSample] = useState<PreparedSample | null>(null);
  const [sampleMessage, setSampleMessage] = useState("");
  const connectionSequence = useRef(0);
  const modalGpuChosenByUser = useRef(false);
  const remoteDeviceChosenByUser = useRef(false);
  const connectionBusy = modalDeploying || remoteChecking;
  const invalidateConnection = () => {
    connectionSequence.current += 1;
    setRemote(null);
    setRemoteMessage("");
    setRemoteChecking(false);
    setModalWorkspaceIssue(false);
  };
  const set = <K extends keyof RunFormState>(key: K, value: RunFormState[K]) => {
    if (key === "remote_hardware") remoteDeviceChosenByUser.current = true;
    if (["execution_target", "remote_endpoint", "modal_gpu"].includes(key)) {
      invalidateConnection();
    }
    setForm((current) => ({ ...current, [key]: value }));
  };

  const refreshExecutionConfigurations = async () => {
    const result = await getJson<{ items: ExecutionConfiguration[] }>("/api/execution-configurations");
    if (!result || !Array.isArray(result.items) || !result.items.every((item) =>
      item && typeof item.name === "string" && item.settings && typeof item.settings === "object" && !Array.isArray(item.settings))) {
      throw new Error("Unable to load saved execution configurations: the server returned an invalid configuration list.");
    }
    setExecutionConfigurations(result.items);
  };
  useEffect(() => { void refreshExecutionConfigurations().catch((reason) => setExecutionMessage(errorMessage(reason))); }, []);
  const loadExecution = (name: string) => {
    if (connectionBusy) return;
    const configuration = executionConfigurations.find((item) => item.name === name);
    setSelectedExecution(name);
    if (!configuration) return;
    loadedExecution.current = true;
    invalidateConnection();
    setExecutionName(name);
    setForm((current) => {
      const next = { ...current, ...configuration.settings };
      if (next.execution_target === "remote") next.remote_hardware = "cuda:0";
      const changedOrigin = ["execution_target", "remote_endpoint"].some((key) => current[key as keyof RunFormState] !== next[key as keyof RunFormState]);
      return { ...next, ...(changedOrigin ? { modal_api_key: "" } : {}),
        num_retry_cluster_connection: configuration.settings.num_retry_cluster_connection ?? "", cluster_config: "" };
    });
    setExecutionMessage(`Loaded ${name}. Edits below apply to the next run; Save updates the saved copy.`);
  };
  const saveExecution = async () => {
    setExecutionBusy(true);
    try {
      if (form.cluster_config) throw new Error("Clear the expert Cluster config path before saving. Saved configurations use the visible machine controls.");
      const settings: Record<string, unknown> = Object.fromEntries(EXECUTION_KEYS.map((key) => [key, form[key]]));
      for (const key of ["num_workers", "threads_per_worker", "dask_scheduler_port"]) settings[key] = Number(settings[key]);
      settings.num_retry_cluster_connection = form.num_retry_cluster_connection === "" ? null : Number(form.num_retry_cluster_connection);
      const saved = await getJson<ExecutionConfiguration>("/api/execution-configurations", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: executionName.trim(), settings }),
      });
      await refreshExecutionConfigurations();
      setSelectedExecution(saved.name);
      setExecutionName(saved.name);
      setExecutionMessage(`Saved ${saved.name} on this Studio host. Credentials are not stored.`);
    } catch (reason) { setExecutionMessage(errorMessage(reason)); }
    finally { setExecutionBusy(false); }
  };
  const deleteExecution = async () => {
    setExecutionBusy(true);
    try {
      await getJson(`/api/execution-configurations/${encodeURIComponent(selectedExecution)}`, { method: "DELETE" });
      await refreshExecutionConfigurations();
      setSelectedExecution(""); setExecutionName("");
      setExecutionMessage("Saved configuration deleted. Current controls are unchanged.");
    } catch (reason) { setExecutionMessage(errorMessage(reason)); }
    finally { setExecutionBusy(false); }
  };
  const updateHydraOverride = (
    setter: Dispatch<SetStateAction<HydraOverrideMap>>,
    key: string,
    value: string,
  ) => setter((current) => {
    if (!value) {
      const next = { ...current };
      delete next[key];
      return next;
    }
    return { ...current, [key]: value };
  });

  useEffect(() => {
    if (!hardware || loadedExecution.current || form.dask_worker_hosts.trim()) return;
    const splatDevice = hardware.devices.find((item) => item.supports_gaussian_splatting);
    setForm((current) => ({
      ...current,
      hardware: splatDevice?.id ?? hardware.devices[0]?.id ?? "cpu",
      splat_implementation: current.execution_target === "local" && !splatDevice ? "none" : current.splat_implementation,
    }));
  }, [hardware]);

  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams({
      config_name: form.config_name,
      loader: form.loader,
      gaussian_splatting_config_name: form.gaussian_splatting_config_name,
      splat_implementation: form.splat_implementation,
    });
    setHydraParametersLoading(true);
    setHydraParametersError("");
    void getJson<HydraParameterSchema>(`/api/configuration/parameters?${query}`, { signal: controller.signal })
      .then((parameters) => {
        setHydraParameters(parameters);
        const modelKeys = new Set(parameters.model.fields.map((field) => field.key));
        const gaussianKeys = new Set(parameters.gaussian.fields.map((field) => field.key));
        setModelHydraOverrides((current) => Object.fromEntries(Object.entries(current).filter(([key]) => modelKeys.has(key))));
        setSplatHydraOverrides((current) => Object.fromEntries(Object.entries(current).filter(([key]) => gaussianKeys.has(key))));
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setHydraParameters(null);
        setHydraParametersError(errorMessage(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setHydraParametersLoading(false);
      });
    return () => controller.abort();
  }, [form.config_name, form.loader, form.gaussian_splatting_config_name, form.splat_implementation, hydraParametersRevision]);

  const modelCatalog = remote?.configuration?.models ?? schema.models;
  const model = modelCatalog.find((item) => item.id === form.config_name);
  const capabilities = model?.capabilities ?? { iterative_splat: false, mvs: false, share_intrinsics: false };
  const splatOptions = useMemo(() => schema.splat_implementations.map((item) => ({
    ...item, status: item.id === "gsplat" && !capabilities.iterative_splat ? "disabled" : "available",
  })), [schema, capabilities.iterative_splat]);
  const splatMeta = schema.splat_implementations.find((item) => item.id === form.splat_implementation);
  const selectedHardware = hardware?.devices.find((item) => item.id === form.hardware);
  const hasExternalWorkers = form.execution_target === "local" && Boolean(form.dask_worker_hosts.trim());
  const modalWorkspaceSelected = form.execution_target === "remote";
  const selectedRemoteHardware = remote?.hardware.devices.find((item) => item.id === form.remote_hardware);
  const deviceError = modalWorkspaceSelected && form.remote_hardware !== "cuda:0"
    ? "This Modal workspace runs on cuda:0. Select its GPU device before starting."
    : !hasExternalWorkers && form.execution_target === "remote" && remote && (!selectedRemoteHardware || (selectedRemoteHardware.status && selectedRemoteHardware.status !== "available"))
      ? "Choose an available device on the remote machine before starting."
      : !hasExternalWorkers && form.execution_target === "local" && hardware && (!selectedHardware || (selectedHardware.status && selectedHardware.status !== "available"))
        ? "Choose an available device on this machine before starting."
        : "";

  useEffect(() => {
    if (!modalWorkspaceSelected || loadedExecution.current || remoteDeviceChosenByUser.current) return;
    setForm((current) => current.remote_hardware === "cuda:0" ? current : { ...current, remote_hardware: "cuda:0" });
  }, [modalWorkspaceSelected]);
  const machineProfile = advancedMachineProfile(form, hardware, selectedHardware);
  const selectedSample = samples.find((item) => item.id === form.sample_id);
  const datasetAnalysis = inputMode === "sample"
    ? preparedSample?.analysis
    : inputMode === "previous" ? previousImages?.analysis ?? previousFolder?.analysis
    : imagesFolder?.analysis?.image_count ? imagesFolder.analysis : datasetFolder?.analysis;
  const modalGpuRecommendation = recommendModalGpu(form, datasetAnalysis, selectedSample?.image_count ?? 0);
  const modalRecommendationKey = [
    inputMode,
    form.sample_id,
    datasetAnalysis?.image_count ?? 0,
    datasetAnalysis?.average_megapixels ?? 0,
    datasetAnalysis?.total_megapixels ?? 0,
    form.max_resolution,
    form.config_name,
    form.splat_implementation,
    form.run_mvs,
  ].join(":");
  const modalGpuOptions = MODAL_GPU_CHOICES.map((item) => ({
    ...item,
    label: item.id === modalGpuRecommendation?.gpu.id ? `${item.label} · Suggested` : item.label,
  }));
  const useDeployedModalWorkspace = (gpu: string) => {
    const gpuInfo = MODAL_GPU_CHOICES.find((item) => item.id === gpu);
    const workspace: RemoteWorkspace = {
      configuration: schema,
      verified: false,
      hardware: {
        summary: `Modal ${gpu} workspace`,
        devices: [{
          id: "cuda:0",
          kind: "cuda",
          label: `Modal ${gpu}`,
          details: "NVIDIA CUDA GPU · starts with the first reconstruction",
          memory: gpuInfo ? `${gpuInfo.memoryGiB} GB` : undefined,
          supports_gaussian_splatting: true,
        }],
      },
    };
    setRemote(workspace);
    return workspace;
  };

  useEffect(() => {
    if (form.execution_target !== "remote") return;
    if (!modalGpuRecommendation || loadedExecution.current || modalGpuChosenByUser.current || connectionBusy) return;
    if (form.modal_gpu === modalGpuRecommendation.gpu.id) return;
    set("modal_gpu", modalGpuRecommendation.gpu.id);
    if (form.remote_endpoint) {
      setRemote(null);
      setModalWorkspaceIssue(true);
      setRemoteMessage("The dataset changed the recommended VM. Update the Modal workspace to apply it, then verify readiness.");
    }
  }, [modalGpuRecommendation?.gpu.id, modalRecommendationKey, form.modal_gpu, form.remote_endpoint, form.execution_target, connectionBusy]);

  useEffect(() => {
    if (loadedExecution.current || hasExternalWorkers) return;
    setForm((current) => ({
      ...current,
      num_workers: machineProfile.workers,
      threads_per_worker: storedMachineThreads(machineProfile),
      worker_memory_limit: `${machineProfile.memoryPerWorkerGiB}GB`,
    }));
  }, [machineProfile.key]);

  useEffect(() => {
    if (!selectedSample || preparedSample?.path !== "") return;
    setSampleMessage(form.execution_target === "remote"
      ? "Will download directly on Modal"
      : selectedSample.prepared ? "Cached and ready" : "Will download when the run starts");
  }, [form.execution_target, selectedSample, preparedSample?.path]);

  const updateModalGpu = (value: string) => {
    modalGpuChosenByUser.current = true;
    set("modal_gpu", value);
    setRemote(null);
    setModalWorkspaceIssue(Boolean(form.remote_endpoint));
    setRemoteMessage("VM selection changed. Update the Modal workspace to apply it, then verify readiness.");
  };

  useEffect(() => {
    if (form.splat_implementation === "gsplat" && model && !capabilities.iterative_splat) set("splat_implementation", "none");
  }, [form.splat_implementation, model, capabilities.iterative_splat]);

  useEffect(() => {
    if (form.share_intrinsics && model && !capabilities.share_intrinsics) set("share_intrinsics", false);
  }, [form.share_intrinsics, model, capabilities.share_intrinsics]);

  async function inspectRemote(endpoint: string, apiKey: string, sequence = connectionSequence.current) {
    const payload = await getJson<RemoteWorkspace>("/api/remote/inspect", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint, api_key: apiKey }) });
    if (sequence === connectionSequence.current) setRemote({ ...payload, verified: true });
    return payload;
  }

  const deployModal = async () => {
    if (account?.status !== "connected") throw new Error("Connect your Modal account before running.");
    const sequence = connectionSequence.current;
    setModalDeploying(true);
    setModalWorkspaceIssue(false);
    setRemote(null);
    setRemoteMessage("");
    try {
      let deployment = await getJson<ModalDeployment>("/api/modal/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: account.profile,
          force: modalWorkspaceIssue,
          gpu: form.modal_gpu,
          cpu: Math.max(1, Number(form.num_workers) * Number(form.threads_per_worker) || 1),
          memory_mb: Math.max(4096, Math.round(memoryGiB(form.worker_memory_limit) * Math.max(1, Number(form.num_workers) || 1) * 1024)),
        }),
      });
      if (sequence !== connectionSequence.current) return;
      setModalDeployment(deployment);
      while (["queued", "running", "cancelling"].includes(deployment.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        deployment = await getJson<ModalDeployment>(`/api/modal/deploy/${encodeURIComponent(deployment.id)}`);
        if (sequence !== connectionSequence.current) return;
        setModalDeployment(deployment);
      }
      if (deployment.status === "cancelled") {
        setRemoteMessage("Modal workspace setup stopped.");
        return;
      }
      if (deployment.status === "failed") throw new Error(deployment.error || "Modal deployment failed");
      setForm((current) => ({ ...current, remote_endpoint: deployment.endpoint, modal_api_key: deployment.api_key }));
      useDeployedModalWorkspace(deployment.gpu);
      setModalDeployment({
        ...deployment,
        status: "running",
        phase: "verifying",
        stage: "Starting and verifying the Modal workspace",
        log_tail: [...deployment.log_tail, `Registered endpoint ${deployment.endpoint}`, "Checking the CPU control service; the GPU remains off until a run starts…"],
      });
      setRemoteChecking(true);
      try {
        await inspectRemote(deployment.endpoint, deployment.api_key, sequence);
        if (sequence !== connectionSequence.current) return;
      } catch (reason) {
        if (sequence !== connectionSequence.current) return;
        const message = `Deployment finished, but the workspace health check failed: ${errorMessage(reason)}`;
        setRemote((current) => current ? { ...current, verified: false } : null);
        setModalWorkspaceIssue(true);
        setModalDeployment({
          ...deployment,
          status: "failed",
          phase: "verifying",
          stage: "Modal workspace needs an update",
          error: message,
          log_tail: [...deployment.log_tail, `Registered endpoint ${deployment.endpoint}`, message],
        });
        throw new Error(message);
      } finally {
        if (sequence === connectionSequence.current) setRemoteChecking(false);
      }
      setModalDeployment({
        ...deployment,
        phase: "ready",
        stage: "Modal workspace ready",
        log_tail: [
          ...deployment.log_tail,
          `Registered endpoint ${deployment.endpoint}`,
          "Workspace health check passed. The Modal GPU is ready.",
        ],
      });
      setModalWorkspaceIssue(false);
      setRemoteMessage(`Modal ${deployment.gpu} workspace passed its health check and is ready to run.`);
      return deployment;
    } catch (reason) {
      if (sequence !== connectionSequence.current) return;
      const message = errorMessage(reason);
      setModalWorkspaceIssue(true);
      setRemoteMessage(/Request failed \(404\)/.test(message)
        ? "This GTSFM server was started before Modal deployment support was installed. Stop it with Ctrl-C, run `gtsfm run` again, then click Deploy."
        : message);
    } finally {
      setModalDeploying(false);
    }
  };

  const stopModalDeployment = async () => {
    if (!modalDeployment || !["queued", "running"].includes(modalDeployment.status)) return;
    try {
      const deployment = await getJson<ModalDeployment>(`/api/modal/deploy/${encodeURIComponent(modalDeployment.id)}/cancel`, { method: "POST" });
      setModalDeployment(deployment);
      setRemoteMessage("Stopping Modal workspace setup…");
    } catch (reason) {
      setRemoteMessage(errorMessage(reason));
    }
  };

  const chooseSample = (sampleId: string) => {
    const sample = samples.find((item) => item.id === sampleId);
    setPreparedSample(null);
    setSampleMessage("");
    if (!sample) {
      setForm((current) => ({ ...current, sample_id: "", dataset_dir: "" }));
      return;
    }
    const recommendations = sample.recommendations;
    setLoaderOptions(recommendations.loader_options ?? {});
    setForm((current) => ({
      ...current,
      sample_id: sample.id,
      dataset_dir: "",
      images_dir: "",
      name: sample.id,
      loader: recommendations.loader,
      config_name: recommendations.config_name,
      max_resolution: recommendations.max_resolution ?? current.max_resolution,
    }));
    setPreparedSample({
      path: "",
      sample,
      analysis: { image_count: sample.image_count, image_bytes: 0, total_megapixels: 0, average_megapixels: 0, max_width: 0, max_height: 0 },
    });
    setSampleMessage(sample.prepared ? "Cached and ready" : "Will download where the run executes");
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submissionPending.current) return;
    submissionPending.current = true;
    setBusy(true); setError("");
    try {
      if (deviceError) throw new Error(deviceError);
      let endpoint = form.remote_endpoint, apiKey = form.modal_api_key;
      if (modalWorkspaceSelected) {
        const deployment = await deployModal();
        if (!deployment) throw new Error("Modal preparation did not finish. Check its status below and retry.");
        endpoint = deployment.endpoint; apiKey = deployment.api_key;
      }
      const expertOverrides = form.advanced_overrides.split("\n");
      const payload = { ...form, remote_endpoint: endpoint,
        loader: form.loader,
        hydra_overrides: modelHydraOverrides,
        advanced_overrides: expertOverrides.map((line) => line.trim()).filter(Boolean),
        gaussian_splatting_overrides: splatHydraOverrides,
        api_key: form.execution_target === "remote" ? apiKey : "", loader_options: loaderOptions,
        hardware: form.execution_target === "remote" ? form.remote_hardware : form.hardware,
        max_run_minutes: Number(form.max_run_minutes),
        max_resolution: form.max_resolution ? Number(form.max_resolution) : null,
        num_workers: Number(form.num_workers), threads_per_worker: Number(form.threads_per_worker),
        gs_max_steps: Number(form.gs_max_steps), live_preview_interval: Number(form.live_preview_interval),
        max_frame_lookahead: form.max_frame_lookahead ? Number(form.max_frame_lookahead) : null,
        num_matched: form.num_matched ? Number(form.num_matched) : null,
        num_retry_cluster_connection: form.num_retry_cluster_connection ? Number(form.num_retry_cluster_connection) : null };
      const job = await getJson<Job>("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      onStarted(job);
    } catch (reason) { setError(errorMessage(reason)); } finally { submissionPending.current = false; setBusy(false); }
  };

  const sourceReady = Boolean(form.dataset_dir || (inputMode === "sample" && preparedSample));
  const imageCount = inputMode === "sample" ? preparedSample?.sample.image_count || 0 : datasetAnalysis?.image_count || 0;
  const sourceName = inputMode === "sample" ? samples.find((item) => item.id === form.sample_id)?.label : inputMode === "previous" ? previousFolder?.name : datasetFolder?.name;
  const computeName = form.execution_target === "local" ? selectedHardware?.label || "This machine" : `Modal ${form.modal_gpu}`;
  useEffect(() => {
    if (!reuse) return;
    const settings = reuse.settings;
    const sample = samples.find((item) => item.id === settings.sample_id);
    if (sample) { setInputMode("sample"); chooseSample(sample.id); }
    else { setInputMode("upload"); setDatasetFolder(null); setImagesFolder(null); setPreparedSample(null); }
    const excluded = new Set<string>([...EXECUTION_KEYS, "sample_id", "dataset_dir", "images_dir", "cluster_config"]);
    const allowed = Object.fromEntries(Object.entries(settings).filter(([key]) =>
      key in EMPTY_FORM && !excluded.has(key) && !key.startsWith("modal_")));
    setForm((current) => ({
      ...current,
      ...allowed,
      sample_id: sample?.id ?? "",
      dataset_dir: sample ? "" : String(settings.dataset_dir ?? ""),
      images_dir: sample ? "" : String(settings.images_dir ?? ""),
      name: `${String(settings.name || "experiment")}-copy`,
      advanced_overrides: Array.isArray(settings.advanced_overrides)
        ? settings.advanced_overrides.join("\n")
        : typeof settings.advanced_overrides === "string" ? settings.advanced_overrides : "",
    }));
    setLoaderOptions((settings.loader_options || {}) as LoaderValues);
    setModelHydraOverrides((settings.hydra_overrides || {}) as HydraOverrideMap);
    setSplatHydraOverrides((settings.gaussian_splatting_overrides || {}) as HydraOverrideMap);
    setReuseMessage("Settings loaded. Review the images and compute, then run when ready.");
    const controller = new AbortController();
    if (!sample && settings.dataset_dir) {
      void getJson<{items: UploadedFolder[]}>("/api/uploads", {signal: controller.signal}).then(({items}) => {
        if (controller.signal.aborted || currentForm.current.dataset_dir !== settings.dataset_dir) return;
        setDatasetFolder(items.find((item) => item.path === settings.dataset_dir) || null);
        setImagesFolder(items.find((item) => item.path === settings.images_dir) || null);
      }).catch(() => {});
    }
    return () => controller.abort();
  }, [reuse?.version]);

  return <form id="runForm" onSubmit={submit}>
    <div className="run-page-heading">
      <div><span className="eyebrow">GTSfM experiment workspace</span><h1>New experiment</h1><p>Choose images, a preset, and compute. Inspect and reuse the result here.</p></div>
      {schemaLoading ? <small className="catalog-sync"><RefreshCw className="spin" size={12}/> Loading options</small>
        : schemaError ? <button className="catalog-sync failed" type="button" onClick={onRetrySchema}><CircleAlert size={12}/> Options offline · Retry</button>
        : <button className="catalog-sync" type="button" onClick={onRetrySchema}><RefreshCw size={12}/> Refresh options</button>}
    </div>
    {reuseMessage && <p className="reuse-notice" role="status">{reuseMessage}</p>}
    <fieldset className="run-body" disabled={busy}><div className="run-fields">
    <Section number="01" title="Add your images" subtitle="" active>
      <TextField label="Experiment name" value={form.name} onChange={(value) => set("name", value)} autoComplete="off" />
      <div className="segmented input-source" role="group" aria-label="Input source">
        <button type="button" className={`target-choice ${inputMode === "upload" ? "active" : ""}`} onClick={() => {
          setInputMode("upload"); setLoaderOptions({});
          setForm((current) => ({ ...current, sample_id: "", dataset_dir: datasetFolder?.path ?? "", images_dir: imagesFolder?.path ?? "" }));
        }}><FolderUp size={13} /> Upload your own</button>
        <button type="button" className={`target-choice ${inputMode === "sample" ? "active" : ""}`} onClick={() => {
          setInputMode("sample");
          const recommendations = preparedSample?.sample.recommendations;
          setLoaderOptions(recommendations?.loader_options ?? {});
          setForm((current) => ({ ...current, sample_id: preparedSample?.sample.id ?? "", dataset_dir: preparedSample?.path ?? "", images_dir: "", loader: recommendations?.loader ?? current.loader }));
        }}><Database size={13} /> Sample datasets</button>
        <button type="button" className={`target-choice ${inputMode === "previous" ? "active" : ""}`} onClick={() => {
          setInputMode("previous"); setLoaderOptions({}); setError("");
          setForm((current) => ({ ...current, sample_id: "", dataset_dir: previousFolder?.path ?? "", images_dir: previousImages?.path ?? "", loader: inputMode === "sample" ? EMPTY_FORM.loader : current.loader }));
          void refreshUploads();
        }}><FolderUp size={13} /> Previous uploads</button>
      </div>
      {inputMode === "upload" ? <>
        <FolderDropField label="Image or dataset folder" value={datasetFolder} onError={setError} onUploaded={(folder) => {
          setDatasetFolder(folder); set("dataset_dir", folder?.path ?? "");
          if (folder && form.name === "my-scene") set("name", folder.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "my-scene");
        }} />
        <details className="disclosure source-details"><summary>Separate image folder or dataset options</summary><FolderDropField label="Separate images folder" optional value={imagesFolder} onError={setError} onUploaded={(folder) => {
          setImagesFolder(folder); set("images_dir", folder?.path ?? "");
        }} /></details>
      </> : inputMode === "previous" ? <div className="sample-picker">
        <SelectField label="Previously uploaded folder" value={previousFolder?.id ?? ""}
          options={previousUploads.map((folder) => ({ id: folder.id, label: `${folder.name} · ${folder.analysis.image_count} image${folder.analysis.image_count === 1 ? "" : "s"} · ${folder.uploaded_at ? new Date(folder.uploaded_at * 1000).toLocaleString() : folder.id}` }))}
          empty={uploadsLoading ? "Loading previous uploads…" : "Choose an uploaded folder…"} disabled={uploadsLoading}
          onChange={(id) => {
            const folder = previousUploads.find((item) => item.id === id) ?? null;
            setPreviousFolder(folder); setPreviousImages(null); setError("");
            setForm((current) => ({ ...current, sample_id: "", dataset_dir: folder?.path ?? "", images_dir: "", name: folder?.name ?? current.name }));
          }} />
        {previousFolder && <p className="field-help">{previousFolder.file_count.toLocaleString()} file{previousFolder.file_count === 1 ? "" : "s"} · {formatBytes(previousFolder.bytes)} · Ready to reuse</p>}
        <SelectField label="Separate uploaded images folder (optional)" value={previousImages?.id ?? ""}
          options={previousUploads.map((folder) => ({ id: folder.id, label: `${folder.name} · ${folder.id}` }))}
          empty="Use images from the dataset folder" disabled={uploadsLoading}
          onChange={(id) => { const folder = previousUploads.find((item) => item.id === id) ?? null; setPreviousImages(folder); set("images_dir", folder?.path ?? ""); }} />
        {!uploadsLoading && !uploadsError && !previousUploads.length && <p className="field-help">No previous uploads yet. Use “Upload your own” to add a folder.</p>}
        {uploadsError && <p className="job-error" role="alert">{uploadsError}</p>}
        <button type="button" className="text-button" disabled={uploadsLoading} onClick={() => void refreshUploads()}><RefreshCw size={12} /> Refresh uploads</button>
      </div> : <div className="sample-picker">
        <SelectField label="Sample scene" value={form.sample_id} options={samples} empty={samplesLoading ? "Loading Sample datasets…" : "Choose a GTSFM sample…"} onChange={chooseSample} disabled={samplesLoading} />
        {form.sample_id && <div className={`sample-card ${preparedSample ? "ready" : ""}`}>
          <span className="sample-state">{preparedSample ? <Check size={14} /> : <Database size={14} />}</span>
          <div><strong>{samples.find((item) => item.id === form.sample_id)?.label}</strong><small>{samples.find((item) => item.id === form.sample_id)?.description}</small>
            <span>{sampleMessage} · {samples.find((item) => item.id === form.sample_id)?.image_count} images · <a href={samples.find((item) => item.id === form.sample_id)?.source_url} target="_blank" rel="noreferrer">View on GitHub ↗</a></span></div>
        </div>}
        <p className="field-help sample-help">Dataset format, VGGT model, resolution, and available loader settings are applied automatically.</p>
      </div>}
      {inputMode === "sample" && form.sample_id === "lund-door" && <div className="capture-preview" aria-label="Lund Door source photographs">{[1,6,12].map((index) => <img key={index} src={`/static/samples/lund-door/${index}.jpg`} alt={`Lund Door photograph ${index}`}/>)}</div>}
      <details className="disclosure dataset-options"><summary>Dataset format & loader options <span>{displayName(form.loader)}</span></summary><SelectField
        label="Dataset format"
        value={form.loader}
        options={schema.loaders}
        onChange={(value) => {
          set("loader", value);
          setLoaderOptions({});
        }}
      />
      <LoaderOptions descriptors={schema.loader_options[form.loader] ?? []} values={loaderOptions} setValues={setLoaderOptions} /></details>
      {form.dataset_dir && !datasetFolder && inputMode === "upload" && <p className="reuse-notice">Reusing {form.dataset_dir.split(/[\\/]/).filter(Boolean).at(-1)}. Choose a folder to replace it.</p>}
    </Section>
    <Section number="02" title="Choose your pipeline" subtitle="Start with a preset. Fine-tune only what your scene needs." active>
      <SelectField label="Reconstruction model" value={form.config_name} options={modelCatalog.filter((item) => ["vggt", "fastvggt", "sift_front_end", form.config_name].includes(item.id))} onChange={(value) => { set("config_name", value); setModelHydraOverrides({}); }} />
      <SelectField label="Splat implementation" value={form.splat_implementation} options={splatOptions} onChange={(value) => set("splat_implementation", value)} />
      <p className="field-help">{splatMeta?.description}</p>
      {form.splat_implementation === "gsplat" && <div className="nested-options">
        <SelectField label="Optimizer preset" value={form.gaussian_splatting_config_name} options={remote?.configuration?.gaussian_splatting_models ?? schema.gaussian_splatting_models} onChange={(value) => { set("gaussian_splatting_config_name", value); setSplatHydraOverrides({}); }} />
      </div>}
      <Collapsible.Root className="advanced-options configuration-advanced" open={configurationAdvanced} onOpenChange={setConfigurationAdvanced}>
        <Collapsible.Trigger className="advanced-trigger"><span><SlidersHorizontal size={13} /><span className="advanced-trigger-copy">Advanced Configuration<small>{Object.keys(modelHydraOverrides).length + Object.keys(splatHydraOverrides).length} Hydra overrides</small></span></span><ChevronDown size={14} /></Collapsible.Trigger>
        <Collapsible.Content className="advanced-content">
          <SelectField label="All reconstruction presets" value={form.config_name} options={modelCatalog} onChange={(value) => {set("config_name", value); setModelHydraOverrides({});}}/><p className="configuration-intro">Tune reconstruction and splat optimization without editing YAML. Empty fields inherit the selected preset.</p>
          <div className="configuration-json-action">
            <button type="button" onClick={() => setJsonConfigurationOpen(true)} disabled={!hydraParameters || hydraParametersLoading}>{"{ }"} Paste JSON configuration</button>
            <small>Import a preset-aware JSON object instead of filling fields individually.</small>
          </div>
          {form.splat_implementation === "gsplat" && <>
            <div className="form-grid configuration-core-fields">
              <TextField label="Training steps" type="number" min="1" step="1" value={form.gs_max_steps} onChange={(value) => set("gs_max_steps", value)} />
              <TextField label="Preview every" type="number" min="10" step="10" value={form.live_preview_interval} onChange={(value) => set("live_preview_interval", value)} />
            </div>
          </>}
          <div className="form-grid">
            <TextField label="Maximum image resolution" type="number" min="1" value={form.max_resolution} onChange={(value) => set("max_resolution", value)} placeholder="Model default" />
            <SelectField label="Graph partitioner" value={form.graph_partitioner} options={schema.graph_partitioners} empty="Model default" onChange={(value) => set("graph_partitioner", value)} />
            <SelectField label="Global descriptor" value={form.global_descriptor_config_name} options={schema.global_descriptors} empty="Model default" onChange={(value) => set("global_descriptor_config_name", value)} />
            <SelectField label="Image retriever" value={form.retriever_config_name} options={schema.retrievers} empty="Model default" onChange={(value) => set("retriever_config_name", value)} />
            <SelectField label="Correspondence" value={form.correspondence_generator_config_name} options={schema.correspondence_generators} empty="Model default" onChange={(value) => set("correspondence_generator_config_name", value)} />
            <SelectField label="Verifier" value={form.verifier_config_name} options={schema.verifiers} empty="Model default" onChange={(value) => set("verifier_config_name", value)} />
            <TextField label="Frame lookahead" type="number" min="0" value={form.max_frame_lookahead} onChange={(value) => set("max_frame_lookahead", value)} placeholder="Model default" />
            <TextField label="Matches per image" type="number" min="0" value={form.num_matched} onChange={(value) => set("num_matched", value)} placeholder="Model default" />
          </div>
          {capabilities.share_intrinsics && <Toggle id="shareIntrinsics" checked={form.share_intrinsics} onCheckedChange={(value) => set("share_intrinsics", value)}>Share camera intrinsics</Toggle>}
          {hydraParametersLoading && <p className="configuration-unavailable"><RefreshCw className="spin" size={10} /> Loading the selected presets in the background…</p>}
          {hydraParametersError && <p className="configuration-unavailable configuration-error">Could not load preset controls. {hydraParametersError} <button type="button" onClick={() => setHydraParametersRevision((value) => value + 1)}>Retry</button></p>}
          {hydraParameters && <>
            {hydraParameters.model.fields.length
              ? <HydraFieldGroups fields={hydraParameters.model.fields} values={modelHydraOverrides} onChange={(key, value) => updateHydraOverride(setModelHydraOverrides, key, value)} />
              : <p className="configuration-unavailable">The selected reconstruction preset has no additional safe scalar settings.</p>}
            {form.splat_implementation === "gsplat" && <HydraFieldGroups fields={hydraParameters.gaussian.fields} values={splatHydraOverrides} onChange={(key, value) => updateHydraOverride(setSplatHydraOverrides, key, value)} />}
          </>}
          <Field label="Expert Hydra overrides" optional><textarea rows={4} value={form.advanced_overrides} onChange={(event) => set("advanced_overrides", event.target.value)} placeholder="One key=value override per line" /></Field>
        </Collapsible.Content>
      </Collapsible.Root>
      {jsonConfigurationOpen && hydraParameters && <ConfigurationJsonDialog
        parameters={hydraParameters}
        model={modelHydraOverrides}
        gaussian={splatHydraOverrides}
        expert={form.advanced_overrides}
        onApply={(configuration) => {
          setModelHydraOverrides(configuration.model);
          setSplatHydraOverrides(configuration.gaussian);
          set("advanced_overrides", configuration.expert.join("\n"));
        }}
        onClose={() => setJsonConfigurationOpen(false)}
      />}
    </Section>
    </div><aside className="compute-column">
    <Section number="03" title="Choose where to run" subtitle="Use this machine or connect your own remote compute." active>
      <div className="compute-choices" id="computeTarget" role="group" aria-label="Compute destination">
        {[{id: "local", label: "This computer", detail: "Use the hardware available here", icon: <Cpu size={22}/>}, {id: "modal", label: "Modal cloud", detail: "Connect once. GPU setup is automatic.", icon: <Server size={22}/>}].map((choice) => <button type="button" key={choice.id} aria-pressed={choice.id === (form.execution_target === "local" ? "local" : "modal")} disabled={busy || connectionBusy} onClick={() => { invalidateConnection(); setForm((current) => ({...current, execution_target: choice.id === "local" ? "local" : "remote", remote_hardware: choice.id === "modal" ? "cuda:0" : current.remote_hardware})); }}>{choice.icon}<span><strong>{choice.label}</strong><small>{choice.detail}</small></span><span className="choice-dot"/></button>)}
      </div>
      {form.execution_target === "local" && form.splat_implementation !== "none" && hardware && !hardware.devices.some((device) => device.supports_gaussian_splatting && (!device.status || device.status === "available")) && <HardwareWarning hardware={hardware} onUseRemote={() => set("execution_target", "remote")}/>}
      {hasExternalWorkers && <>
        <TextField label="Worker device" value={form.execution_target === "local" ? form.hardware : form.remote_hardware} onChange={(value) => set(form.execution_target === "local" ? "hardware" : "remote_hardware", value)} placeholder="cpu, cuda:0, or rocm:0" />
        <p className="field-help">Select the device on each SSH worker. The control machine can be CPU-only.</p>
      </>}
      {form.execution_target === "local" ? <>
        {!hasExternalWorkers && (hardware ? <><SelectField label="Hardware" value={form.hardware} options={hardware.devices} onChange={(value) => set("hardware", value)} /><HardwareCard device={selectedHardware} /></> : <div className="hardware-card detecting"><RefreshCw className="spin" size={13}/><div><strong>Detecting compute devices…</strong><small>You can configure the rest of the run while this finishes.</small></div></div>)}
      </> : <div className="nested-options remote-vm-options">
          <div className="provider-panel modal-account">
            <h3>{account?.status === "connected" ? `Connected to ${account.workspace || account.profile}` : "Connect your Modal account"}</h3>
            <p>{account?.status === "connected" ? "Your native Modal profile is ready. Run will build and check the workspace before starting compute." : "Sign in through Modal in your browser. Credentials stay on this Studio host."}</p>
            {account?.profiles && account.profiles.length > 1 && <SelectField label="Modal account" value={account.profile} options={account.profiles.map((name) => ({id: name, label: name}))} empty="Choose a saved account" onChange={(value) => { invalidateConnection(); void connectAccount(false, value); }}/>}
            {account?.status !== "connected" && <button type="button" className="secondary-action full-width" disabled={accountBusy || account?.status === "signing_in"} onClick={() => void connectAccount()}>{accountBusy ? "Checking account…" : account?.status === "signing_in" ? "Waiting for browser sign-in…" : account?.profiles.length ? "Connect saved Modal account" : "Sign in to Modal"}</button>}
            {account?.login_url && <a className="primary-action" href={account.login_url} target="_blank" rel="noreferrer">Continue to Modal ↗</a>}
            {account?.code && <p>Verification code: <strong>{account.code}</strong></p>}
            {account?.error && <p className="job-error" role="alert">{account.error}</p>}
            {account?.status === "error" && <button type="button" className="text-button" onClick={() => void connectAccount(true)}>Sign in again in browser</button>}
            <SelectField label="GPU" value={form.modal_gpu} options={modalGpuOptions} onChange={updateModalGpu} disabled={busy || connectionBusy}/><p className="field-help">The suggested GPU is a starting point based on image count and settings. Increase memory if your scene needs it.</p>
            <p className="field-help">One GPU per run. Your Modal account is billed for compute used. Stop the experiment here to cancel it.</p>
            {modalDeployment && <ModalDeploymentProgress deployment={modalDeployment} onCancel={stopModalDeployment} onExpand={() => setModalLogExpanded(true)}/>}
            {modalDeployment && modalLogExpanded && <ModalDeploymentDialog deployment={modalDeployment} onCancel={stopModalDeployment} onClose={() => setModalLogExpanded(false)}/>}
            {remoteMessage && <p className="field-help" role="status">{remoteMessage}</p>}
          </div>
      </div>}
    <TextField label="Run time limit (minutes)" type="number" min="1" max="1440" value={form.max_run_minutes} onChange={(value) => set("max_run_minutes", value)}/>
    <p className="field-help">After setup, requests cancellation when this limit is reached. Modal bills for actual usage.</p>
    <details className="disclosure saved-execution"><summary>Saved compute settings</summary>
      <SelectField label="Saved execution configuration" value={selectedExecution} options={executionConfigurations.map((item) => ({ id: item.name, label: item.name }))} empty="Current settings (unsaved)" onChange={loadExecution} disabled={executionBusy || connectionBusy} />
      {executionMessage.startsWith("Unable") && <p className="job-error" role="alert">{executionMessage}</p>}
      <details className="disclosure saved-settings"><summary>Save or manage this configuration</summary><TextField label="Configuration name" value={executionName} onChange={setExecutionName} placeholder="Mac, Cyber, or Cluster" />
      <div className="job-actions">
        <button type="button" className="secondary-action" disabled={executionBusy || !executionName.trim()} onClick={saveExecution}>Save configuration</button>
        <button type="button" className="text-button" disabled={executionBusy || connectionBusy || !selectedExecution} onClick={() => loadExecution(selectedExecution)}>Reload saved</button>
        <button type="button" className="text-button danger" disabled={executionBusy || !selectedExecution} onClick={deleteExecution}>Delete saved</button>
      </div>
      <p className="field-help" role="status">{executionMessage.startsWith("Unable") ? "" : executionMessage || "Saved on this Studio host. Edits below are used for this run."}</p></details>
    </details>
    <Collapsible.Root className="advanced-options" open={advanced} onOpenChange={setAdvanced}>
      <Collapsible.Trigger className="advanced-trigger"><span><Cpu size={13} /><span className="advanced-trigger-copy">Machine settings<small>{machineProfile.label}</small></span></span><ChevronDown size={14} /></Collapsible.Trigger>
      <Collapsible.Content className="advanced-content">
        <div className="machine-profile-summary">
          <div><span>MACHINE PROFILE</span><strong>{machineProfile.label}</strong></div>
          <p>{machineProfile.description}</p>
          <div className="machine-profile-specs"><span>{form.num_workers} worker{Number(form.num_workers) === 1 ? "" : "s"}</span><span>{form.threads_per_worker} thread{Number(form.threads_per_worker) === 1 ? "" : "s"} / worker</span><span>{form.worker_memory_limit} / worker</span></div>
        </div>
        <div className="form-grid">
          <TextField label="Workers" type="number" min="1" value={form.num_workers} onChange={(value) => set("num_workers", value)} disabled={!machineProfile.allowWorkers} title={!machineProfile.allowWorkers ? "Fixed for the selected single-GPU machine" : undefined} />
          <TextField label="Threads / worker" type="number" min="1" value={form.threads_per_worker} onChange={(value) => {
            set("threads_per_worker", value);
            rememberMachineThreads(machineProfile, value);
          }} disabled={!machineProfile.allowThreads} />
          <TextField label="Memory / worker" value={form.worker_memory_limit} onChange={(value) => set("worker_memory_limit", value)} disabled={!machineProfile.allowMemory} />
        </div>
        {!machineProfile.allowWorkers && <p className="advanced-machine-note">Worker count is locked because the selected machine exposes one GPU. Choose a CPU machine to distribute work across multiple workers.</p>}
        <div className="form-grid">
          <SelectField label="Log level" value={form.log} options={schema.log_levels} onChange={(value) => set("log", value)} />
          <TextField label="Dashboard port" value={form.dashboard_port} onChange={(value) => set("dashboard_port", value)} placeholder=":8787" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Input worker" value={form.input_worker} onChange={(value) => set("input_worker", value)} placeholder="Optional worker address" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Dask temp folder" value={form.dask_tmpdir} onChange={(value) => set("dask_tmpdir", value)} placeholder="System default" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Cluster config (YAML)" value={form.cluster_config} onChange={(value) => set("cluster_config", value)} placeholder="Optional YAML path" disabled={!machineProfile.allowLocalRuntime} aria-describedby={machineProfile.allowLocalRuntime ? "cluster-config-help" : undefined} />
          <TextField label="Cluster retries" type="number" min="0" value={form.num_retry_cluster_connection} onChange={(value) => set("num_retry_cluster_connection", value)} placeholder="3" disabled={!machineProfile.allowLocalRuntime} />
        </div>
        {machineProfile.allowLocalRuntime && <p id="cluster-config-help" className="field-help">Path on this Studio host to a Dask constructor YAML or existing cluster file. A LocalCluster constructor file controls cluster options instead of the worker, thread, and memory defaults above.</p>}
        {machineProfile.allowLocalRuntime && <>
          <TextField label="SSH worker hosts" value={form.dask_worker_hosts} onChange={(value) => set("dask_worker_hosts", value)} placeholder="Optional: worker1 worker2" />
          <TextField label="Scheduler host" value={form.dask_scheduler_host} onChange={(value) => set("dask_scheduler_host", value)} placeholder="First worker host" />
          <TextField label="Scheduler port" type="number" min="0" max="65535" value={form.dask_scheduler_port} onChange={(value) => set("dask_scheduler_port", value)} placeholder="0 = automatic" />
          <TextField label="Worker Python" value={form.dask_worker_python} onChange={(value) => set("dask_worker_python", value)} placeholder="Same Python path as the runner" />
          <p className="field-help">Leave worker hosts empty to compute on the selected machine. With hosts set, native Dask SSHCluster starts the scheduler and workers from that machine using its SSH configuration. Nodes need the same GTSFM environment and shared input paths. Workers and memory above apply per host.</p>
        </>}
      </Collapsible.Content>
    </Collapsible.Root>
    </Section>
    </aside></fieldset>
    <footer className="run-footer">
      <div className="run-footer-copy"><strong>{sourceReady ? `${imageCount ? `${imageCount} images · ` : ""}${model?.label || displayName(form.config_name)}` : "Add images to begin"}</strong><span>{sourceReady ? `${computeName} · ${form.splat_implementation === "none" ? "Cameras and point cloud" : "Reconstruction and Gaussian splats"}` : "Choose a folder, a sample, or a previous upload above."}</span></div>
      {modalDeploying && modalDeployment && <div className="preparing-actions"><button type="button" className="text-button" onClick={() => setModalLogExpanded(true)}>View setup logs</button><button type="button" className="text-button danger" disabled={modalDeployment.phase === "verifying"} onClick={stopModalDeployment}>Stop setup</button></div>}
      <button className="primary-action" type="submit" disabled={!sourceReady || busy || connectionBusy || Boolean(deviceError) || (form.execution_target === "remote" && account?.status !== "connected")}>{busy ? <RefreshCw className="spin" size={20}/> : <Play size={20}/>}<span>{busy ? (modalDeploying ? "Preparing Modal…" : "Starting experiment…") : "Run experiment"}</span></button>
      {(error || deviceError) && <div className="form-error" role="alert">{error || deviceError}</div>}
    </footer>
  </form>;
}

const statusLabel = (status: JobStatus): string => ({ queued: "Queued", running: "Running", completed: "Completed", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted" })[status];
const canRetryJob = (job: Job): boolean => ["failed", "interrupted"].includes(job.status) && job.retry_available !== false;

function SplatDownload({ jobId, compact = false }: { jobId: string; compact?: boolean }) {
  const href = `/api/jobs/${encodeURIComponent(jobId)}/splat?format=ply`;
  return <div className={`splat-download ${compact ? "compact" : ""}`}>
    <span className="splat-format-label">.PLY</span>
    <a href={href} download title="Download PLY splat"><Download size={12}/>{!compact && <span>Download</span>}</a>
  </div>;
}

interface ActivityPanelProps {
  onRetried: (job: Job) => void;
  jobs: Job[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onBrowse: () => void;
  onCancel: (id: string) => void;
  onRefresh: () => void;
}

function CopyDiagnosticPrompt({ jobId }: { jobId: string }) {
  const [state, setState] = useState<"idle" | "loading" | "copied" | "error">("idle");
  const [error, setError] = useState("");
  useEffect(() => {
    if (state !== "copied") return;
    const timer = window.setTimeout(() => setState("idle"), 2000);
    return () => window.clearTimeout(timer);
  }, [state]);

  const copyPrompt = async () => {
    setState("loading");
    setError("");
    try {
      const { prompt } = await getJson<{ prompt: string }>(`/api/jobs/${encodeURIComponent(jobId)}/diagnostic-prompt`);
      if (!await copyTextToClipboard(prompt)) throw new Error("Clipboard access failed. Please try copying again.");
      setState("copied");
    } catch (reason) {
      setError(errorMessage(reason));
      setState("error");
    }
  };

  return <span className="diagnostic-prompt-action">
    <button className="text-button" type="button" onClick={copyPrompt} disabled={state === "loading"} title="Copy a diagnosis request with the CLI command and full run log">
      {state === "copied" ? <Check size={12}/> : <Copy size={12}/>}
      {state === "loading" ? "Preparing prompt…" : state === "copied" ? "Copied!" : "Copy diagnostic prompt"}
    </button>
    {error && <span className="job-error" role="alert">{error}</span>}
  </span>;
}

function RetryJob({ jobId, onStarted, compact = false }: { jobId: string; onStarted: (job: Job) => void; compact?: boolean }) {
  const pending = useRef(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const retry = async () => {
    if (pending.current) return;
    pending.current = true;
    setStarting(true); setError("");
    try {
      const job = await getJson<Job>(`/api/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" });
      onStarted(job);
    } catch (reason) { setError(errorMessage(reason)); }
    finally { pending.current = false; setStarting(false); }
  };
  return <span className="retry-job-action">
    <button className={compact ? "secondary-action" : "text-button"} type="button" onClick={retry} disabled={starting} title="Start a new run with these settings and the current code">
      <RefreshCw size={12} className={starting ? "spin" : undefined}/>{starting ? "Retrying…" : "Retry"}
    </button>
    {error && <span className="job-error" role="alert">{error}</span>}
  </span>;
}

function ActivityPanel({ jobs, activeId, onSelect, onBrowse, onCancel, onRefresh, onRetried }: ActivityPanelProps) {
  return <div><div className="panel-title"><div><h1>Experiments</h1><p>Settings, progress, and results together.</p></div><button className="icon-button" title="Refresh" onClick={onRefresh}><RefreshCw size={14} /></button></div>
    <button type="button" className="text-button existing-results" onClick={onBrowse}>Open existing reconstruction</button><div className="job-list">{jobs.length ? jobs.map((job) => <article key={job.id} className={`job-card ${job.id === activeId ? "selected" : ""}`}>
      <div className="job-card-top"><strong>{job.name}</strong><span className={`job-status ${job.status}`}>{statusLabel(job.status)}</span></div>
      <small>{displayName(job.spec.config_name || "GTSFM")} · {displayName(job.spec.splat_implementation || "no_splats")}</small>
      {job.error && <p className="job-error">{job.error}</p>}
      {job.created_at && <time className="job-created" dateTime={job.created_at}>{new Date(job.created_at).toLocaleString()}</time>}
      {job.status === "failed" && <CopyDiagnosticPrompt key={job.id} jobId={job.id} />}
      <div className="job-actions">{canRetryJob(job) && <RetryJob jobId={job.id} onStarted={onRetried}/>}<button className="text-button" onClick={() => onSelect(job.id)}>{job.has_final_splat ? "View splat" : ["queued", "running"].includes(job.status) ? "View progress" : "View details"}</button>
        {["queued", "running"].includes(job.status) && <button className="text-button danger" onClick={() => onCancel(job.id)}><Square size={9} fill="currentColor"/> Stop</button>}
        {job.status === "completed" && job.output_root && <><a className="text-button" href={`/api/jobs/${encodeURIComponent(job.id)}/results`} download>Download results</a></>}
        {job.remote?.workspace_url && <a className="text-button" href={job.remote.workspace_url} target="_blank" rel="noreferrer">Remote results ↗</a>}
      </div>{job.has_final_splat && <SplatDownload jobId={job.id} compact />}</article>) : <div className="empty-state"><Activity size={20} /><strong>No runs yet</strong><span>Start with New experiment.</span></div>}</div>
  </div>;
}

function ResultsPanel({ onStarted, refreshKey, onSceneOpened }: { onStarted: (job: Job) => void; refreshKey: string; onSceneOpened: () => void }) {
  const [scene, setScene] = useState<ResultItem | null>(null);
  const [steps, setSteps] = useState("7000");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!scene || starting) return;
    setStarting(true); setError("");
    try {
      const job = await getJson<Job>("/api/scenes/splatify", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene: scene.rel_path, gs_max_steps: Number(steps) }),
      });
      setScene(null); onStarted(job);
    } catch (reason) { setError(errorMessage(reason)); }
    finally { setStarting(false); }
  };
  const runRoot = new URLSearchParams(location.search).get("run");
  return <div><div className="panel-title"><div><span className="eyebrow">SCENE LIBRARY</span><h2>Reconstructions</h2><p>Select a scene to explore.</p></div></div>
    {runRoot && <a className="text-button" href="/?view=results">View all results</a>}
    {scene && <form className="splat-result-form" onSubmit={start}>
      <strong>Splatify this reconstruction</strong>
      <small className="result-path">{scene.rel_path}</small>
      <p>{(scene.point_count ?? 0).toLocaleString()} points · {(scene.camera_count ?? 0).toLocaleString()} cameras. Train Gaussian splats from the saved cameras and original photos.</p>
      <TextField label="Training steps" type="number" min="1" value={steps} onChange={setSteps} />
      {error && <p className="job-error" role="alert">{error}</p>}
      <div className="job-actions"><button className="primary-action" type="submit" disabled={starting || !Number.isInteger(Number(steps)) || Number(steps) < 1}>{starting ? "Starting…" : "Start splatting"}</button><button className="text-button" type="button" disabled={starting} onClick={() => setScene(null)}>Cancel</button></div>
    </form>}
    <ResultsBrowser refreshKey={refreshKey} onOpened={onSceneOpened} onSplatify={(item) => { setScene(item); setError(""); }}/>

  </div>;
}

interface StatusBarProps {
  onRetried: (job: Job) => void;
  job: Job | null;
  live: LiveState | null;
  logsOpen: boolean;
  setLogsOpen: Dispatch<SetStateAction<boolean>>;
  visible: boolean;
  onCancel: (id: string) => void;
  onClose: () => void;
}

interface StatusBarPosition {
  left: number;
  top: number;
  width: number;
}

function StatusBar({ job, live, logsOpen, setLogsOpen, visible, onCancel, onClose, onRetried }: StatusBarProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef<{
    pointerX: number;
    pointerY: number;
    left: number;
    top: number;
    width: number;
    maxLeft: number;
    maxTop: number;
  } | null>(null);
  const [position, setPosition] = useState<StatusBarPosition | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const bar = barRef.current;
    const container = bar?.parentElement;
    if (!bar || !container || typeof ResizeObserver === "undefined") return;
    const keepInBounds = () => setPosition((current) => {
      if (!current) return null;
      const containerRect = container.getBoundingClientRect();
      const width = Math.min(current.width, containerRect.width);
      const height = bar.getBoundingClientRect().height;
      return {
        left: Math.max(0, Math.min(containerRect.width - width, current.left)),
        top: Math.max(0, Math.min(containerRect.height - height, current.top)),
        width,
      };
    });
    const observer = new ResizeObserver(keepInBounds);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const beginDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest("button, a, input, select")) return;
    const bar = barRef.current;
    const container = bar?.parentElement;
    if (!bar || !container || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const barRect = bar.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const start = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: barRect.left - containerRect.left,
      top: barRect.top - containerRect.top,
      width: barRect.width,
      maxLeft: Math.max(0, containerRect.width - barRect.width),
      maxTop: Math.max(0, containerRect.height - barRect.height),
    };
    dragStart.current = start;
    setPosition({ left: start.left, top: start.top, width: start.width });
    setDragging(true);
  };

  const moveDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = dragStart.current;
    if (!start) return;
    setPosition({
      left: Math.max(0, Math.min(start.maxLeft, start.left + event.clientX - start.pointerX)),
      top: Math.max(0, Math.min(start.maxTop, start.top + event.clientY - start.pointerY)),
      width: start.width,
    });
  };

  const finishDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    dragStart.current = null;
    setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  if (!job || !visible) return null;
  const progress = typeof live?.progress === "number" && Number.isFinite(live.progress) ? live.progress : job.status === "completed" ? 1 : 0;
  const loss = typeof live?.loss === "number" && Number.isFinite(live.loss) ? `loss ${live.loss.toFixed(4)}` : "";
  const splatCount = typeof live?.splat_count === "number" && Number.isFinite(live.splat_count) ? `${live.splat_count.toLocaleString()} splats` : "";
  const stage = job.status === "completed" ? "Reconstruction complete" : job.status === "failed" ? "Reconstruction failed" : job.status === "cancelled" ? "Reconstruction cancelled" : job.status === "interrupted" ? "Previous run outcome unconfirmed" : job.status === "queued" ? "Waiting to start" : live?.stage === "gaussian_splatting" ? "Optimizing Gaussian splats" : "Running reconstruction pipeline";
  const style = position ? { left: position.left, top: position.top, width: position.width, right: "auto", bottom: "auto" } : undefined;
  return <div ref={barRef} className={`run-status-bar ${dragging ? "is-dragging" : ""}`} data-status={job.status} style={style} title="Drag to move run status" onPointerDown={beginDrag} onPointerMove={moveDrag} onPointerUp={finishDrag} onPointerCancel={finishDrag}>
    <div className="status-copy status-drag-handle" onDoubleClick={() => setPosition(null)}><span className="status-dot"/><div><strong>{job.name} · {statusLabel(job.status)}</strong><small>{job.error || stage}</small></div></div>
    <div className="status-metrics"><span>{live?.max_steps ? `${Number(live.step).toLocaleString()} / ${Number(live.max_steps).toLocaleString()} steps` : ""}</span><span>{loss}</span><span>{splatCount}</span></div>
    <div className="status-actions">{canRetryJob(job) && <RetryJob key={job.id} jobId={job.id} onStarted={onRetried} compact/>}{job.has_final_splat && <SplatDownload jobId={job.id} />}{["queued", "running"].includes(job.status) && <button className="secondary-action stop-process" type="button" onClick={() => onCancel(job.id)}><Square size={10} fill="currentColor"/> Stop</button>}<button className="secondary-action" onClick={() => setLogsOpen(!logsOpen)}><Terminal size={13}/> Logs</button><button className="status-close" type="button" title="Close run status" aria-label="Close run status" onClick={onClose}><X size={15}/></button></div>
    <Progress.Root className="run-progress-track" value={progress * 100}><Progress.Indicator className="run-progress-fill" style={{ transform: `translateX(-${100 - progress * 100}%)` }} /></Progress.Root>
  </div>;
}

interface ViewerProps {
  onRetried: (job: Job) => void;
  activeJob: Job | null;
  live: LiveState | null;
  logsOpen: boolean;
  setLogsOpen: Dispatch<SetStateAction<boolean>>;
  statusBarOpen: boolean;
  setStatusBarOpen: Dispatch<SetStateAction<boolean>>;
  onCancelJob: (id: string) => void;
}

// Preserve argument boundaries when copying a subprocess argv into a POSIX shell.
function shellCommand(args: string[]): string {
  return args.map((arg) => /^[A-Za-z0-9_@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, "'\\''")}'`).join(" ");
}

function LogPanel({ open, job, onClose, onRetried, live }: { open: boolean; job: Job | null; onClose: () => void; onRetried: (job: Job) => void; live?: LiveState | null }) {
  const copyReset = useRef<number | null>(null);
  const [copied, setCopied] = useState<"logs" | "command" | null>(null);
  const lines = job?.log_tail || [];
  const remoteRequest = job?.command?.[0] === "remote";
  const command = remoteRequest ? `POST ${job?.command?.[1] || ""}` : shellCommand(job?.command || []);

  const copy = async (kind: "logs" | "command") => {
    if (!await copyTextToClipboard(kind === "command" ? command : lines.join("\n"))) return;
    setCopied(kind);
    if (copyReset.current !== null) window.clearTimeout(copyReset.current);
    copyReset.current = window.setTimeout(() => setCopied(null), 1000);
  };

  useEffect(() => () => { if (copyReset.current !== null) window.clearTimeout(copyReset.current); }, []);
  useEffect(() => { setCopied(null); }, [job?.id]);

  if (!open) return null;
  return <section className="log-drawer" aria-label="Run details">
    <div className="log-header"><strong><Terminal size={14}/> Run details</strong><div className="log-header-actions"><button type="button" title="Close run details" aria-label="Close run details" onClick={onClose}><X size={16}/></button></div></div>
    {job && <div className="run-details-summary"><strong>{job.name}</strong><span className={`job-status ${job.status}`}>{statusLabel(job.status)}</span>{job.return_code != null && <span>Exit code {job.return_code}</span>}{job.output_root && <div className="run-output-path"><span>Output</span><code>{job.output_root}</code></div>}{job.error && <p className="job-error">{job.error}</p>}</div>}
    {job?.history_note && <p className="job-history-note run-history-note">{job.history_note}</p>}
    {job && ["failed", "interrupted"].includes(job.status) && <div className="run-diagnostic-prompt">{canRetryJob(job) && <RetryJob key={`retry-${job.id}`} jobId={job.id} onStarted={onRetried}/>}<CopyDiagnosticPrompt key={job.id} jobId={job.id} /></div>}
    <section className="run-command" aria-label={remoteRequest ? "Remote request" : "CLI command"}>
      <div className="run-detail-heading"><strong>{remoteRequest ? "Remote request" : "CLI command"}</strong><button type="button" disabled={!command} onClick={() => copy("command")} aria-label={remoteRequest ? "Copy remote request" : "Copy command"}>{copied === "command" ? <Check size={12}/> : <Copy size={12}/>} {copied === "command" ? "Copied!" : "Copy"}</button></div>
      <pre>{command || "No command recorded for this run."}</pre>
    </section>
    {live?.message && <p className="inspector-stage" role="status">{live.message}</p>}
    <div className="run-detail-heading log-section-heading"><strong>Logs</strong><button type="button" onClick={() => copy("logs")} aria-label="Copy logs">{copied === "logs" ? <Check size={12}/> : <Copy size={12}/>} {copied === "logs" ? "Copied!" : "Copy"}</button></div>
    <pre id="runLogs">{lines.join("\n") || "No log output yet."}</pre>
  </section>;
}

function SetupPanel({ setup, refreshing, onRefresh, onSetupChange, onClose }: {
  setup: SetupStatus | null;
  refreshing: boolean;
  onRefresh: () => void;
  onSetupChange: (setup: SetupStatus) => void;
  onClose: () => void;
}) {
  const [installingId, setInstallingId] = useState<string | null>(null);
  const [actionErrors, setActionErrors] = useState<Record<string, string>>({});

  const runSetupAction = async (item: SetupCheck) => {
    if (!item.action?.enabled || installingId) return;
    setInstallingId(item.id);
    setActionErrors((current) => ({ ...current, [item.id]: "" }));
    try {
      const payload = await getJson<{ setup: SetupStatus }>(`/api/setup/${encodeURIComponent(item.id)}/install`, { method: "POST" });
      onSetupChange(payload.setup);
    } catch (reason) {
      setActionErrors((current) => ({ ...current, [item.id]: errorMessage(reason) }));
    } finally {
      setInstallingId(null);
    }
  };

  const status = setup?.status ?? "warning";
  const optionalUnavailable = setup?.counts.optional ?? 0;
  return <section className="setup-panel" aria-label="Setup checks" aria-live="polite">
    <header className="setup-panel-header">
      <span className={`setup-overall-icon ${status}`} aria-hidden="true">{status === "ready" ? <CheckCircle2 size={16} /> : <CircleAlert size={16} />}</span>
      <div className="setup-panel-copy"><span>SETUP CHECKS</span><strong>{setup?.summary ?? "Checking environment…"}</strong></div>
      <div className="setup-panel-actions">
        <button type="button" title="Refresh checks" aria-label="Refresh setup checks" onClick={onRefresh} disabled={refreshing}><RefreshCw className={refreshing ? "spin" : ""} size={13} /></button>

        <button type="button" title="Close checks" aria-label="Close setup checks" onClick={onClose}><X size={14} /></button>
      </div>
    </header>
    <div className="setup-panel-body">
      {!setup ? <div className="setup-loading"><RefreshCw className="spin" size={13} /> Inspecting this machine…</div> : <>
        <div className="setup-summary-line"><span>{setup.counts.ready} passed</span>{optionalUnavailable > 0 && <span>{optionalUnavailable} optional unavailable</span>}</div>
        <ul>{setup.items.map((item) => <li key={item.id} data-state={item.state}>
          <span className="setup-check-mark" aria-hidden="true">{item.state === "ready" ? <Check size={12} /> : item.state === "error" ? <X size={12} /> : <span />}</span>
          <div className="setup-check-content"><div className="setup-check-title"><strong>{item.label}</strong><span className="setup-check-tools">{!item.required && <em>optional</em>}{item.action && <button className="setup-install" type="button" disabled={!item.action.enabled || Boolean(installingId)} title={item.action.reason || item.action.label} onClick={() => runSetupAction(item)}>{installingId === item.id ? <RefreshCw className="spin" size={9}/> : item.action.enabled ? <Download size={9}/> : null}<span>{installingId === item.id ? "Working…" : item.action.label}</span></button>}</span></div><small className={actionErrors[item.id] ? "setup-action-error" : undefined}>{actionErrors[item.id] || item.detail}</small></div>
        </li>)}</ul>
      </>}
    </div>
  </section>;
}

function DaskStats({ job, live }: { job: Job; live: LiveState | null }) {
  const stats = live?.dask;
  const memoryRatio = stats?.memory_limit_bytes ? Math.min(1, stats.memory_bytes / stats.memory_limit_bytes) : 0;
  const stage = live?.stage === "gaussian_splatting" ? "Optimizing splats" : job.status === "queued" ? "Waiting to start" : "Reconstructing scene";
  return <section className="dask-stats" aria-label="Live Dask status">
    <div className="dask-stats-heading"><div><span>DASK LIVE</span><strong>{stage}</strong></div>{stats?.dashboard_url && <a href={stats.dashboard_url} target="_blank" rel="noreferrer">Open dashboard ↗</a>}</div>
    {!stats ? <div className="dask-stats-waiting"><span className="dask-pulse"/> Starting workers…</div> : <>
      <div className="dask-stat-grid">
        <div><span>Workers</span><strong>{stats.workers}</strong><small>{stats.threads} threads</small></div>
        <div><span>Running</span><strong>{stats.running_tasks}</strong><small>{stats.pending_tasks} pending</small></div>
        <div><span>Finished</span><strong>{stats.completed_tasks}</strong><small>{stats.failed_tasks ? `${stats.failed_tasks} failed` : "no errors"}</small></div>
        <div><span>CPU</span><strong>{Math.round(stats.cpu_percent)}%</strong><small>across workers</small></div>
      </div>
      <div className="dask-memory"><div><span>Memory</span><strong>{formatBytes(stats.memory_bytes)} / {formatBytes(stats.memory_limit_bytes)}</strong></div><div className="dask-memory-track"><span style={{ width: `${memoryRatio * 100}%` }}/></div></div>
    </>}
  </section>;
}

function pipelineStatusMessage(job: Job, live: LiveState | null): string {
  if (live?.stage === "gaussian_splatting") {
    const step = Number(live.step || 0);
    const maxSteps = Number(live.max_steps || 0);
    return maxSteps > 0
      ? `GTSFM: Optimizing Gaussian splats · ${step.toLocaleString()} / ${maxSteps.toLocaleString()} steps`
      : "GTSFM: Initializing Gaussian optimization…";
  }
  if (live?.message) return live.message;
  if (job.status === "queued") return job.remote ? "Waiting for the Modal GPU worker…" : "Waiting for the reconstruction worker…";
  const latestStatus = [...(job.log_tail || [])].reverse().find((line) => /GTSFM|partition|VGGT|Gaussian|splat/i.test(line));
  return latestStatus?.replace(/^.*?\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL):\s*/, "") || "GTSFM: Preparing the reconstruction pipeline…";
}

function Viewer({ activeJob, live, logsOpen, setLogsOpen, statusBarOpen, setStatusBarOpen, onCancelJob, onRetried }: ViewerProps) {
  const closeStatus = () => { setStatusBarOpen(false); setLogsOpen(false); };
  const showDaskStats = Boolean(activeJob && ["queued", "running"].includes(activeJob.status));
  const expectsSplats = Boolean(activeJob && (activeJob.spec.splat_implementation || "none") !== "none");
  const visualizationAvailable = Boolean(live?.preview_url || live?.final_url || activeJob?.has_final_splat);
  const showPipelineWait = Boolean(activeJob && ["queued", "running"].includes(activeJob.status) && !visualizationAvailable);
  return <div className="experiment-visualization"><main id="main-content"><StatusBar job={activeJob} live={live} logsOpen={logsOpen} setLogsOpen={setLogsOpen} visible={statusBarOpen} onCancel={onCancelJob} onClose={closeStatus} onRetried={onRetried} />
    <div className="viewer-top-panels">
    <SceneStats>{showDaskStats && activeJob && <DaskStats job={activeJob} live={live}/>}</SceneStats>
    <ViewControls/>
    </div>
    <ViewerCanvas/>
    <div id="viewer-startup-anchor"/>
    <ViewerToolbar/>
    {showPipelineWait && activeJob && <div className="pipeline-wait-overlay" role="status" aria-live="polite"><div className="pipeline-wait-content"><span>RECONSTRUCTION IN PROGRESS</span><strong>{pipelineStatusMessage(activeJob, live)}</strong><div className="pipeline-wait-dots" aria-hidden="true"><i/><i/><i/></div><small>{expectsSplats ? "The first live Gaussian preview will appear here automatically." : "The reconstruction and source photographs will appear here automatically."}</small></div></div>}
    {!showPipelineWait && <ViewerEmpty/>}
    <ViewerLoading/>
  </main><PhotoPanel/></div>;
}

function hardwareWarningContent(hardware: HardwareCatalog): { title: string; description: string } {
  const available = hardware.devices.filter((device) => !device.status || device.status === "available");
  const apple = hardware.devices.find((device) => device.kind === "mps");
  if (apple) return {
    title: "Gaussian splatting is unavailable on this Mac",
    description: "Reconstruction works on this Mac. Gaussian splatting requires an NVIDIA GPU; Apple Metal (MPS) is not supported for splats.",
  };

  const amd = hardware.devices.find((device) => device.kind === "rocm" || /\b(amd|radeon|rocm)\b/i.test(`${device.label} ${device.details}`));
  if (amd) return {
    title: "AMD ROCm is not supported",
    description: `${amd.label} was detected. AMD ROCm can run reconstruction, but GTSFM Gaussian splatting currently requires NVIDIA CUDA. Use a Remote VM with an NVIDIA GPU to generate splats.`,
  };

  const nvidia = hardware.devices.find((device) => device.kind === "nvidia" || device.kind === "cuda" || /\bnvidia\b/i.test(device.label));
  if (nvidia) return {
    title: "NVIDIA GPU found, but CUDA is not ready",
    description: `${nvidia.label} was detected, but this PyTorch environment cannot currently use it for Gaussian splatting. Install a CUDA-enabled PyTorch setup, or use a Remote VM with NVIDIA CUDA.`,
  };

  const otherAccelerator = available.find((device) => device.kind !== "cpu");
  if (otherAccelerator) return {
    title: `${otherAccelerator.label} is not supported for splats`,
    description: "This accelerator can still be used where supported for reconstruction, but GTSFM Gaussian splatting currently requires NVIDIA CUDA. Use a compatible Remote VM to generate splats.",
  };

  return {
    title: "No supported GPU was detected",
    description: "This machine can run CPU reconstruction, but GTSFM Gaussian splatting currently requires an NVIDIA CUDA GPU. Use a Remote VM to generate splats.",
  };
}

function HardwareWarning({ hardware, onUseRemote }: { hardware: HardwareCatalog; onUseRemote: () => void }) {
  const content = hardwareWarningContent(hardware);
  const acknowledgmentKey = `gtsfm.hardware-notice.v1:${content.title}`;
  const [acknowledged, setAcknowledged] = useState(() => {
    try { return Boolean(localStorage.getItem(acknowledgmentKey)); }
    catch { return false; }
  });
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === acknowledgmentKey) setAcknowledged(Boolean(event.newValue));
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [acknowledgmentKey]);
  const acknowledge = () => {
    setAcknowledged(true);
    try { localStorage.setItem(acknowledgmentKey, new Date().toISOString()); }
    catch { /* Keep the notice dismissed for this session when storage is unavailable. */ }
  };
  if (acknowledged) return null;
  return <section className="hardware-warning" role="note" aria-labelledby="hardware-warning-title" aria-describedby="hardware-warning-description">
    <div className="hardware-warning-icon" aria-hidden="true"><CircleAlert size={18}/></div>
    <div className="hardware-warning-copy">
      <strong id="hardware-warning-title">{content.title}</strong>
      <p id="hardware-warning-description">{content.description}</p>
      <div className="hardware-warning-actions"><button type="button" onClick={acknowledge} title="Remember my acknowledgment and don't show this notice again">Got it</button><button type="button" onClick={() => { acknowledge(); onUseRemote(); }}><Server size={12}/> Use Remote VM</button></div>
    </div>
  </section>;
}

const SIDEBAR_WIDTH_KEY = "gtsfm-studio-library-width";
const SIDEBAR_MIN_WIDTH = 280;
const SIDEBAR_MAX_WIDTH = 520;

function constrainSidebarWidth(width: number): number {
  const viewerRoom = Math.max(SIDEBAR_MIN_WIDTH, window.innerWidth - 600);
  return Math.round(Math.min(SIDEBAR_MAX_WIDTH, viewerRoom, Math.max(SIDEBAR_MIN_WIDTH, width)));
}

function storedSidebarWidth(): number {
  try {
    const saved = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return constrainSidebarWidth(Number.isFinite(saved) && saved > 0 ? saved : 320);
  } catch {
    return 320;
  }
}

function App() {
  const { viewer } = useViewerApi();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [environmentOpen, setEnvironmentOpen] = useState(false);
  const [compactNavigation, setCompactNavigation] = useState(() => window.matchMedia("(max-width: 580px)").matches);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 580px)");
    const update = () => setCompactNavigation(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!environmentOpen) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Element && !event.target.closest('.environment-popover, .environment-button')) setEnvironmentOpen(false);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") { setEnvironmentOpen(false); document.querySelector<HTMLButtonElement>('.environment-button')?.focus(); } };
    window.addEventListener("pointerdown", dismiss);
    window.addEventListener("keydown", escape);
    return () => { window.removeEventListener("pointerdown", dismiss); window.removeEventListener("keydown", escape); };
  }, [environmentOpen]);
  const [sidebarWidth, setSidebarWidth] = useState(storedSidebarWidth);
  const [sidebarResizing, setSidebarResizing] = useState(false);
  const sidebarResizeStart = useRef<{ pointerX: number; width: number; pointerId: number; target: HTMLDivElement } | null>(null);
  const [tab, setTab] = useState(new URLSearchParams(location.search).has("experiment") ? "activity" : ["activity", "results"].includes(new URLSearchParams(location.search).get("view") || "") ? new URLSearchParams(location.search).get("view")! : "run");
  const [schema, setSchema] = useState<ConfigurationSchema>(BOOTSTRAP_SCHEMA);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [schemaError, setSchemaError] = useState("");
  const [hardware, setHardware] = useState<HardwareCatalog | null>(null);
  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [setupRefreshing, setSetupRefreshing] = useState(false);
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [reuse, setReuse] = useState<{settings: Record<string, unknown>; version: number}>();
  const [mobileDetail, setMobileDetail] = useState(new URLSearchParams(location.search).has("experiment"));
  const [preparing, setPreparing] = useState(false);
  const [importError, setImportError] = useState("");
  const importFile = useRef<HTMLInputElement>(null);
  const [activeId, setActiveId] = useState<string | null>(new URLSearchParams(location.search).get("experiment"));
  const [liveEvent, setLiveEvent] = useState<JobEvent | null>(null);
  const [logsOpen, setLogsOpen] = useState(false);
  const [statusBarOpen, setStatusBarOpen] = useState(true);

  const finishSidebarResize = useCallback(() => {
    const start = sidebarResizeStart.current;
    sidebarResizeStart.current = null;
    setSidebarResizing(false);
    document.body.classList.remove("sidebar-is-resizing");
    if (start?.target.hasPointerCapture(start.pointerId)) start.target.releasePointerCapture(start.pointerId);
  }, []);

  const startSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    if (sidebarCollapsed || event.button !== 0) return;
    event.preventDefault();
    sidebarResizeStart.current = { pointerX: event.clientX, width: sidebarWidth, pointerId: event.pointerId, target: event.currentTarget };
    event.currentTarget.setPointerCapture(event.pointerId);
    setSidebarResizing(true);
    document.body.classList.add("sidebar-is-resizing");
  };

  const moveSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = sidebarResizeStart.current;
    if (!start) return;
    if (event.buttons === 0) { finishSidebarResize(); return; }
    setSidebarWidth(constrainSidebarWidth(start.width + event.clientX - start.pointerX));
  };

  const resizeSidebarWithKeyboard = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      setSidebarWidth((current) => constrainSidebarWidth(current + direction * (event.shiftKey ? 40 : 10)));
    } else if (event.key === "Home") {
      event.preventDefault();
      setSidebarWidth(420);
    }
  };

  useEffect(() => () => document.body.classList.remove("sidebar-is-resizing"), []);
  useEffect(() => {
    if (!sidebarResizing) return;
    const visibilityChanged = () => { if (document.hidden) finishSidebarResize(); };
    window.addEventListener("pointerup", finishSidebarResize, true);
    window.addEventListener("pointercancel", finishSidebarResize, true);
    window.addEventListener("blur", finishSidebarResize);
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.removeEventListener("pointerup", finishSidebarResize, true);
      window.removeEventListener("pointercancel", finishSidebarResize, true);
      window.removeEventListener("blur", finishSidebarResize);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [sidebarResizing, finishSidebarResize]);
  useEffect(() => {
    try { window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); } catch { /* storage is optional */ }
  }, [sidebarWidth]);
  useEffect(() => {
    const keepSidebarOnScreen = () => setSidebarWidth((current) => constrainSidebarWidth(current));
    window.addEventListener("resize", keepSidebarOnScreen);
    return () => window.removeEventListener("resize", keepSidebarOnScreen);
  }, []);

  const applyJobsSnapshot = useCallback(({ items = [] }: JobsResponse) => {
    setJobs(items);
    setActiveId((current) => items.some((job) => job.id === current)
      ? current
      : items.find((job) => job.status === "running" || job.status === "queued")?.id ?? items[0]?.id ?? null);
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      applyJobsSnapshot(await getJson<JobsResponse>("/api/jobs"));
    } catch (reason) {
      console.warn("Unable to refresh jobs", reason);
    }
  }, [applyJobsSnapshot]);

  const refreshSetup = useCallback(async () => {
    setSetupRefreshing(true);
    try { setSetup(await getJson<SetupStatus>("/api/setup")); }
    catch (reason) { console.warn("Unable to inspect setup", reason); }
    finally { setSetupRefreshing(false); }
  }, []);

  const loadSchema = useCallback(async () => {
    setSchemaLoading(true);
    setSchemaError("");
    try { setSchema(await getJson<ConfigurationSchema>("/api/configuration")); }
    catch (reason) { setSchemaError(errorMessage(reason)); }
    finally { setSchemaLoading(false); }
  }, []);

  const loadHardware = useCallback(async () => {
    try { setHardware(await getJson<HardwareCatalog>("/api/hardware")); }
    catch (reason) { console.warn("Unable to inspect hardware", reason); }
    finally { void refreshSetup(); }
  }, [refreshSetup]);

  const loadSamples = useCallback(async () => {
    try { const payload = await getJson<SamplesResponse>("/api/samples"); setSamples(payload.items); }
    catch (reason) { console.warn("Unable to load sample catalog", reason); }
    finally { setSamplesLoading(false); }
  }, []);

  useEffect(() => { void loadSchema(); void loadHardware(); void loadSamples(); void refreshJobs(); }, [loadSchema, loadHardware, loadSamples, refreshJobs]);
  useEffect(() => {
    const refresh = () => { if (!document.hidden) void loadSchema(); };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [loadSchema]);
  const activeJob = jobs.find((job) => job.id === activeId) ?? null;

  useEffect(() => { if (activeId) setStatusBarOpen(true); }, [activeId]);

  const jobsConnection = useJobStream<JobsResponse>("/api/events/jobs", applyJobsSnapshot);
  const liveConnection = useJobStream<JobEvent>(activeJob ? `/api/events/jobs/${encodeURIComponent(activeJob.id)}` : null, (payload) => {
    setJobs((current) => current.map((job) => job.id === payload.job.id ? payload.job : job));
    setLiveEvent(payload);
  });
  const live = liveEvent && liveEvent.job.id === activeJob?.id ? liveEvent.live : null;
  const connectionMessage = [jobsConnection, liveConnection].includes("unavailable")
    ? "Live updates unavailable. Reload Studio to reconnect."
    : [jobsConnection, liveConnection].includes("reconnecting") ? "Connection interrupted. Reconnecting to live updates…" : "";

  const previewJobId = tab === "activity" && activeJob && (activeJob.status === "queued" || activeJob.status === "running")
    ? activeJob.id
    : null;
  const previewUrl = live?.preview_url;
  const previewLabel = activeJob ? `${activeJob.name} · live` : "";
  useEffect(() => {
    if (!viewer || !previewJobId || !previewUrl) return;
    const controller = new AbortController();
    const { signal } = controller;
    let retry: number | undefined;
    let failures = 0;

    const showPreview = async () => {
      await waitForViewerIdle(viewer, signal);
      if (signal.aborted) return;
      try {
        const loaded = await viewer.loadSplatsFile({ splatsUrl: previewUrl, label: previewLabel, streamId: previewJobId });
        if (loaded !== false) return;
      } catch (reason) {
        if (!signal.aborted) console.warn("Unable to load live preview", reason);
      }
      if (!signal.aborted && ++failures < 3) {
        retry = window.setTimeout(() => { void showPreview(); }, 1200);
      }
    };
    void showPreview();
    return () => {
      controller.abort();
      window.clearTimeout(retry);
    };
  }, [viewer, previewJobId, previewUrl, previewLabel, live?.preview_version]);

  const changeTab = (value: string) => {
    setTab(value);
    setSidebarCollapsed(false);
    setMobileDetail(false);
    setLogsOpen(false);
    if (value === "activity") {
      history.replaceState(null, "", activeId ? `/?experiment=${encodeURIComponent(activeId)}` : "/?view=activity");
    } else {
      history.replaceState(null, "", value === "results" ? "/?view=results" : "/");
    }
  };

  const selectExperiment = (id: string, showLogs = false) => {
    setActiveId(id);
    setTab("activity");
    setMobileDetail(true);
    setStatusBarOpen(true);
    setLogsOpen(showLogs);
    history.replaceState(null, "", `/?experiment=${encodeURIComponent(id)}`);
  };

  const showStartedJob = (job: Job, showLogs = false) => {
    setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    selectExperiment(job.id, showLogs);
    void refreshJobs();
  };
  const onRetried = (job: Job) => showStartedJob(job, true);

  const cancelJob = async (id: string) => { await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); await refreshJobs(); };
  const activeCount = jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  return <div className={`app-shell workspace-${tab} ${mobileDetail ? "has-selection" : ""} ${sidebarCollapsed ? "library-collapsed" : ""} ${logsOpen && activeJob ? "has-details" : ""}`} style={{ "--library-width": `${sidebarWidth}px` } as React.CSSProperties}>
    <header className="workspace-header"><Brand collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((current) => !current)}/><span className="header-divider"/><div className="workspace-location">Workspace <ChevronRight size={13}/><strong>{tab === "run" ? "New experiment" : tab === "activity" ? "Experiments" : "Existing reconstructions"}</strong></div><div className="header-actions"><button type="button" className={`environment-button ${setup?.status || "checking"}`} aria-expanded={environmentOpen} onClick={() => setEnvironmentOpen((open) => !open)}><span/>{setup?.status === "ready" ? "This computer ready" : setup?.status === "error" ? "Setup required" : setup ? "This computer checks" : "Checking this computer"}<ChevronDown size={13}/></button><a href="https://github.com/borglab/gtsfm" target="_blank" rel="noreferrer" aria-label="Open GTSFM GitHub repository"><Github size={18}/></a></div></header>
    {environmentOpen && <div className="environment-popover"><SetupPanel setup={setup} refreshing={setupRefreshing} onRefresh={refreshSetup} onSetupChange={setSetup} onClose={() => setEnvironmentOpen(false)}/></div>}
    {importError && <p className="connection-notice" role="alert">{importError}</p>}
    {connectionMessage && <p className="connection-notice workspace-connection" role="status">{connectionMessage}</p>}
    <Tabs.Root className="workspace-tabs" orientation="horizontal" value={tab} onValueChange={changeTab}>
      <nav className="workspace-nav" aria-label="Studio"><Tabs.List className="studio-tabs" aria-label="Workspace sections"><Tabs.Trigger className="studio-tab" value="run"><Plus size={21}/><span>New experiment</span></Tabs.Trigger><Tabs.Trigger className="studio-tab" value="activity"><Activity size={21}/><span>Experiments</span>{activeCount > 0 && <span id="activeJobCount">{activeCount}</span>}</Tabs.Trigger></Tabs.List><div className="experiment-import"><input ref={importFile} type="file" accept=".json,application/json" hidden aria-label="Import experiment record" onChange={async (event) => { const file = event.target.files?.[0]; if (!file) return; setImportError(""); try { const record = JSON.parse(await file.text()); const result = await getJson<{settings: Record<string, unknown>}>("/api/experiments/import", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(record)}); setReuse({settings: result.settings, version: Date.now()}); changeTab("run"); } catch (reason) { setImportError(errorMessage(reason)); } event.target.value = ""; }}/><button type="button" className="text-button" disabled={preparing} onClick={() => importFile.current?.click()}><FolderUp size={18}/> Import experiment</button></div></nav>
      <aside id="sidebar" role={tab === "results" ? "complementary" : "main"} aria-label={tab === "run" ? "New experiment" : tab === "activity" ? "Experiments" : "Existing reconstructions"} className={`${sidebarCollapsed ? "sidebar-collapsed" : ""} ${sidebarResizing ? "sidebar-resizing" : ""}`} style={{ "--sidebar-width": `${sidebarWidth}px` } as React.CSSProperties}>
      <Tabs.Content className="studio-panel" value="run" forceMount><RunForm onPreparing={setPreparing} reuse={reuse} schema={schema} hardware={hardware} samples={samples} samplesLoading={samplesLoading} onStarted={showStartedJob} schemaLoading={schemaLoading} schemaError={schemaError} onRetrySchema={() => { void loadSchema(); void loadSamples(); }}/></Tabs.Content>
      <Tabs.Content className="studio-panel" value="activity" forceMount><ActivityPanel jobs={jobs} activeId={activeId} onSelect={selectExperiment} onBrowse={() => changeTab("results")} onCancel={cancelJob} onRefresh={refreshJobs} onRetried={onRetried}/></Tabs.Content>
      <Tabs.Content className="studio-panel" value="results" forceMount><ResultsPanel onSceneOpened={() => { if (window.matchMedia("(max-width: 850px)").matches) setSidebarCollapsed(true); }} refreshKey={jobs.filter((job) => ["completed", "failed", "cancelled", "interrupted"].includes(job.status)).map((job) => `${job.id}:${job.status}`).join("|")} onStarted={showStartedJob} /></Tabs.Content>
<div className="sidebar-resize-handle" role="separator" aria-label="Resize side panel" aria-orientation="vertical" aria-valuemin={SIDEBAR_MIN_WIDTH} aria-valuemax={SIDEBAR_MAX_WIDTH} aria-valuenow={sidebarWidth} tabIndex={sidebarCollapsed ? -1 : 0} title="Drag to resize side panel" onPointerDown={startSidebarResize} onPointerMove={moveSidebarResize} onPointerUp={finishSidebarResize} onPointerCancel={finishSidebarResize} onLostPointerCapture={finishSidebarResize} onDoubleClick={() => setSidebarWidth(constrainSidebarWidth(320))} onKeyDown={resizeSidebarWithKeyboard}/></aside>
    </Tabs.Root><div className="viewer-workspace"><ExperimentDetails enabled={tab === "activity"} onBack={() => setMobileDetail(false)} job={activeJob} jobs={jobs} onReuse={(settings) => { setReuse({settings, version: Date.now()}); changeTab("run"); }} onCancel={cancelJob} live={live} onLogs={() => setLogsOpen((open) => !open)}><Viewer key={tab === "activity" && activeId ? activeId : "library"} activeJob={tab === "activity" ? activeJob : null} live={tab === "activity" ? live : null} logsOpen={logsOpen} setLogsOpen={setLogsOpen} statusBarOpen={statusBarOpen} setStatusBarOpen={setStatusBarOpen} onCancelJob={cancelJob} onRetried={onRetried}/></ExperimentDetails></div>{logsOpen && activeJob && <div className="run-details-workspace"><LogPanel open={logsOpen} job={activeJob} onClose={() => setLogsOpen(false)} onRetried={onRetried} live={live}/></div>}</div>;
}

const root = document.getElementById("root");
if (!root) throw new Error("GTSFM Studio root element is missing");
flushSync(() => createRoot(root).render(<ViewerProvider><App /></ViewerProvider>));
