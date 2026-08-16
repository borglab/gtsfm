import React, {
  type Dispatch,
  type FormEvent,
  type InputHTMLAttributes,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
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
import { Activity, Box, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, CircleAlert, Copy, Cpu, Database, Download, FolderUp, Github, Maximize2, Minus, MonitorCog, Play, RefreshCw, Server, SlidersHorizontal, Square, Terminal, Wrench, X } from "lucide-react";
import "./styles.css";

type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
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
  remote_connection: "api" | "ssh";
  remote_provider: string;
  remote_endpoint: string;
  modal_api_key: string;
  modal_token_id: string;
  modal_token_secret: string;
  modal_gpu: string;
  ssh_host: string;
  ssh_port: InputValue;
  ssh_username: string;
  ssh_authentication: "agent" | "key";
  ssh_private_key: string;
  ssh_workspace: string;
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
  image_source?: "prebuilt" | "source";
  runtime_image?: string;
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
  { id: "verifying", label: "Start & verify workspace", detail: "Cold-start the GPU and confirm the workspace API is healthy" },
] as const;

function ModalDeploymentProgress({ deployment, onCancel, onExpand }: { deployment: ModalDeployment; onCancel: () => void; onExpand: () => void }) {
  const ready = deployment.phase === "ready";
  const verifying = deployment.phase === "verifying";
  const currentPhase = ready ? "verifying" : deployment.phase ?? "building";
  const currentIndex = Math.max(0, MODAL_WORKSPACE_STEPS.findIndex((step) => step.id === currentPhase));
  const imageDetail = deployment.image_source === "prebuilt"
    ? "Pull the versioned GTSFM runtime; no package installation"
    : "Install and cache the GTSFM environment";
  return <section className={`modal-deployment ${deployment.status}`} aria-label="Modal workspace progress">
    <div className="modal-deployment-heading"><span>{ready ? <Check size={12}/> : deployment.status === "failed" ? <CircleAlert size={12}/> : deployment.status === "cancelled" ? <Minus size={12}/> : <RefreshCw size={12}/>}<strong>{deployment.stage}</strong></span><span className="modal-deployment-actions"><small>{deployment.image_source === "prebuilt" ? "PREBUILT" : "SOURCE"} · {deployment.gpu}</small>{["queued", "running"].includes(deployment.status) && !verifying && <button className="modal-stop-action" type="button" onClick={onCancel} title="Stop Modal setup"><Square size={9} fill="currentColor"/> Stop</button>}<button type="button" onClick={onExpand} title="View all setup logs" aria-label="View all setup logs"><Maximize2 size={12}/></button></span></div>
    <ol className="modal-deployment-steps">
      {MODAL_WORKSPACE_STEPS.map((step, index) => {
        const complete = ready || index < currentIndex;
        const active = !ready && index === currentIndex;
        const failed = active && deployment.status === "failed";
        const cancelled = active && deployment.status === "cancelled";
        return <li key={step.id} data-state={failed ? "failed" : cancelled ? "cancelled" : complete ? "complete" : active ? "active" : "pending"}>
          <span className="modal-step-mark">{complete ? <Check size={10}/> : failed ? <X size={10}/> : cancelled ? <Minus size={10}/> : index + 1}</span>
          <div><strong>{step.label}</strong><small>{step.id === "building" ? imageDetail : step.detail}</small></div>
        </li>;
      })}
    </ol>
    <div className="modal-log-preview-heading"><span>LIVE SETUP LOGS</span><button type="button" onClick={onExpand}><Maximize2 size={10}/> View all logs</button></div>
    <pre>{deployment.log_tail.slice(-6).join("\n") || "Waiting for Modal setup output…"}</pre>
  </section>;
}

type ModalWorkspaceReadiness = "idle" | "found" | "working" | "ready" | "attention";

function ModalWorkspaceStatus({ state, detail }: { state: ModalWorkspaceReadiness; detail: string }) {
  const title = {
    idle: "Workspace setup required",
    found: "Modal workspace found",
    working: "Preparing Modal workspace",
    ready: "Modal workspace ready",
    attention: "Workspace needs attention",
  }[state];
  return <div className={`modal-workspace-status ${state}`} role="status" aria-live="polite">
    <span aria-hidden="true">{state === "working" ? <RefreshCw className="spin" size={13}/> : state === "ready" ? <Check size={13}/> : state === "attention" ? <CircleAlert size={13}/> : <Server size={13}/>}</span>
    <div><strong>{title}</strong><small>{detail}</small></div>
  </div>;
}

interface JobEvent {
  job: Job;
  live: LiveState;
}

interface Job {
  id: string;
  name: string;
  status: JobStatus;
  spec: {
    config_name?: string;
    splat_implementation?: string;
  };
  error?: string;
  remote?: { workspace_url?: string };
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
  format_detection: DatasetFormatDetection;
}

interface DatasetAnalysis {
  image_count: number;
  image_bytes: number;
  total_megapixels: number;
  average_megapixels: number;
  max_width: number;
  max_height: number;
}

interface DatasetFormatDetection {
  loader: string;
  confidence: number;
  reason: string;
  dataset_subdir?: string | null;
  images_dir?: string | null;
  loader_options: LoaderValues;
  alternatives: string[];
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

interface ViewerApi {
  isBusy(): boolean;
  loadSplatsFile(input: { splatsUrl: string; label: string }): Promise<boolean | void>;
}

declare global {
  interface Window {
    gtsfmViewer?: ViewerApi;
  }
}

const EMPTY_FORM: RunFormState = {
  name: "my-scene", sample_id: "", dataset_dir: "", images_dir: "", loader: "olsson", config_name: "vggt",
  splat_implementation: "gsplat", gaussian_splatting_config_name: "base_gs", gs_max_steps: 7000,
  live_preview_interval: 250, run_mvs: false, execution_target: "local", hardware: "cpu",
  remote_connection: "api", remote_provider: "modal", remote_endpoint: "", modal_api_key: "", modal_token_id: "",
  modal_token_secret: "", modal_gpu: "L40S", ssh_host: "", ssh_port: 22, ssh_username: "", ssh_authentication: "agent",
  ssh_private_key: "", ssh_workspace: "", remote_hardware: "", max_resolution: "", num_workers: 1,
  threads_per_worker: 1, worker_memory_limit: "32GB", graph_partitioner: "", global_descriptor_config_name: "",
  retriever_config_name: "", correspondence_generator_config_name: "", verifier_config_name: "",
  max_frame_lookahead: "", num_matched: "", share_intrinsics: false, log: "INFO", dashboard_port: "",
  input_worker: "", dask_tmpdir: "", cluster_config: "", num_retry_cluster_connection: "", advanced_overrides: "",
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

interface ModalCredentials {
  tokenId: string;
  tokenSecret: string;
}

function parseModalTokenCommand(value: string): ModalCredentials | null {
  const readFlag = (flag: string): string => {
    const escaped = flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = value.match(new RegExp(`(?:^|\\s)${escaped}(?:=|\\s+)(?:"([^"]+)"|'([^']+)'|([^\\s]+))`));
    return (match?.[1] ?? match?.[2] ?? match?.[3] ?? "").trim();
  };
  const tokenId = readFlag("--token-id");
  const tokenSecret = readFlag("--token-secret");
  return tokenId && tokenSecret ? { tokenId, tokenSecret } : null;
}

const modalBearerToken = (form: RunFormState): string => form.modal_api_key;

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
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload as T;
}

const websocketUrl = (path: string): string =>
  `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`;

function Brand({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return <header className="studio-header" aria-label="SfM Studio">
    <div className="brand-primary"><img className="brand-logo" src="/static/brand/sfm-logo.png" alt="SfM" /></div>
    <img className="brand-mark" src="/static/brand/bee-favicon.png" alt="" aria-hidden="true" />
    <button className="sidebar-toggle" type="button" onClick={onToggle} title={collapsed ? "Expand side panel" : "Collapse side panel"} aria-label={collapsed ? "Expand side panel" : "Collapse side panel"}>
      {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
    </button>
  </header>;
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
  return <Field label={label}><select id={id} value={value ?? ""} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
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

const MODAL_GPU_PRICING: ModalGpuChoice[] = [
  { id: "T4", label: "NVIDIA T4 · 16 GB · $0.59/hr", pricePerSecond: 0.000164, relativeSpeed: 0.3, memoryGiB: 16 },
  { id: "L4", label: "NVIDIA L4 · 24 GB · $0.80/hr", pricePerSecond: 0.000222, relativeSpeed: 0.5, memoryGiB: 24 },
  { id: "A10", label: "NVIDIA A10 · 24 GB · $1.10/hr", pricePerSecond: 0.000306, relativeSpeed: 0.58, memoryGiB: 24 },
  { id: "L40S", label: "NVIDIA L40S · 48 GB · $1.95/hr", pricePerSecond: 0.000542, relativeSpeed: 1, memoryGiB: 48 },
  { id: "A100-40GB", label: "NVIDIA A100 · 40 GB · $2.10/hr", pricePerSecond: 0.000583, relativeSpeed: 1.18, memoryGiB: 40 },
  { id: "A100-80GB", label: "NVIDIA A100 · 80 GB · $2.50/hr", pricePerSecond: 0.000694, relativeSpeed: 1.3, memoryGiB: 80 },
  { id: "RTX-PRO-6000", label: "NVIDIA RTX PRO 6000 · 96 GB · $3.03/hr", pricePerSecond: 0.000842, relativeSpeed: 1.45, memoryGiB: 96 },
  { id: "H100", label: "NVIDIA H100 · 80 GB · $3.95/hr", pricePerSecond: 0.001097, relativeSpeed: 1.8, memoryGiB: 80 },
  { id: "H200", label: "NVIDIA H200 · 141 GB · $4.54/hr", pricePerSecond: 0.001261, relativeSpeed: 1.95, memoryGiB: 141 },
  { id: "B200", label: "NVIDIA B200 · 180 GB · $6.25/hr", pricePerSecond: 0.001736, relativeSpeed: 2.35, memoryGiB: 180 },
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
  if (estimatedMemoryGiB <= 14 && form.splat_implementation === "none") gpuId = "T4";
  else if (estimatedMemoryGiB <= 22) gpuId = "L4";
  else if (estimatedMemoryGiB <= 44) gpuId = "L40S";
  else if (estimatedMemoryGiB <= 72 && sceneUnits < 350) gpuId = "A100-80GB";
  else if (estimatedMemoryGiB <= 72) gpuId = "H100";
  else if (estimatedMemoryGiB <= 125) gpuId = "H200";
  else gpuId = "B200";

  const gpu = MODAL_GPU_PRICING.find((item) => item.id === gpuId) ?? MODAL_GPU_PRICING[3];
  const scale = imageCount < 40 ? "small" : imageCount < 140 ? "medium" : imageCount < 350 ? "large" : "very large";
  return {
    gpu,
    imageCount,
    effectiveMegapixels,
    estimatedMemoryGiB,
    reason: `${scale} ${imageCount}-image workload at about ${effectiveMegapixels.toFixed(effectiveMegapixels < 10 ? 1 : 0)} MP per image`,
  };
}

function ModalGpuRecommendationCard({ recommendation, selectedGpu, onApply }: { recommendation: ModalGpuRecommendation | null; selectedGpu: string; onApply: () => void }) {
  if (!recommendation) return <div className="modal-gpu-recommendation empty"><strong>VM recommendation</strong><p>Choose a sample or upload images to size the Modal GPU automatically.</p></div>;
  const selected = selectedGpu === recommendation.gpu.id;
  return <section className={`modal-gpu-recommendation ${selected ? "selected" : "overridden"}`} aria-label="Recommended Modal VM size">
    <div><span>RECOMMENDED VM</span><strong>{recommendation.gpu.id} · {recommendation.gpu.memoryGiB} GB</strong></div>
    <p>{recommendation.reason}. Estimated peak GPU memory is approximately {recommendation.estimatedMemoryGiB} GiB, including working headroom.</p>
    {selected ? <small><Check size={11}/> Selected automatically</small> : <button type="button" onClick={onApply}>Use recommended</button>}
  </section>;
}

const formatDuration = (minutes: number): string => minutes < 60 ? `${Math.max(1, Math.round(minutes))} min` : `${(minutes / 60).toFixed(minutes < 120 ? 1 : 0)} hr`;
const formatCost = (cost: number): string => `$${cost < 10 ? cost.toFixed(2) : cost.toFixed(1)}`;

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
  if (form.execution_target === "remote") {
    if (form.remote_connection === "api" && form.remote_provider === "modal") {
      const resources = MODAL_VM_RESOURCES[form.modal_gpu] ?? MODAL_VM_RESOURCES.L40S;
      const gpu = MODAL_GPU_PRICING.find((item) => item.id === form.modal_gpu);
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
    return {
      key: `remote:${form.remote_connection}:${form.remote_provider}`,
      label: form.remote_connection === "ssh" ? "Direct VM (coming soon)" : `${displayName(form.remote_provider)} VM`,
      description: "Machine-level tuning is unavailable until this remote provider is connected.",
      workers: 1,
      threadsPerWorker: 1,
      memoryPerWorkerGiB: 32,
      allowWorkers: false,
      allowThreads: false,
      allowMemory: false,
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
    allowWorkers: !accelerator,
    allowThreads: Boolean(selected),
    allowMemory: Boolean(selected),
    allowLocalRuntime: true,
  };
}

function ModalCostEstimate({ form, analysis, fallbackImageCount }: { form: RunFormState; analysis?: DatasetAnalysis; fallbackImageCount: number }) {
  const imageCount = analysis?.image_count || fallbackImageCount;
  if (!imageCount) return <div className="modal-estimate empty"><strong>Cost estimate</strong><p>Choose a GTSFM sample or upload an image dataset to calculate an estimate.</p></div>;

  const gpu = MODAL_GPU_PRICING.find((item) => item.id === form.modal_gpu) ?? MODAL_GPU_PRICING[3];
  const sourceAverageMegapixels = analysis?.average_megapixels || 8;
  const maxResolution = Number(form.max_resolution);
  const cappedMegapixels = Number.isFinite(maxResolution) && maxResolution > 0 ? (maxResolution ** 2 * 0.75) / 1_000_000 : sourceAverageMegapixels;
  const effectiveAverageMegapixels = Math.min(sourceAverageMegapixels, cappedMegapixels);
  const pixelFactor = Math.max(0.65, Math.min(2.5, Math.sqrt(effectiveAverageMegapixels / 2)));
  const modelFactor = form.config_name.includes("fast") ? 0.75 : form.config_name === "vggt" ? 1 : 1.35;
  const reconstructionMinutes = (2 + imageCount * 0.09 * pixelFactor + Math.pow(imageCount, 1.35) * 0.025) * modelFactor;
  const steps = Math.max(1, Number(form.gs_max_steps) || 7000);
  const splatMinutes = form.splat_implementation === "gsplat" ? (steps / 1000) * (0.45 + Math.sqrt(imageCount) * 0.08) * pixelFactor : form.splat_implementation === "anysplat" ? 1.5 + imageCount * 0.04 * pixelFactor : 0;
  const mvsMinutes = form.run_mvs ? 1 + imageCount * 0.12 * pixelFactor : 0;
  const referenceMinutes = reconstructionMinutes + splatMinutes + mvsMinutes;
  const estimatedMinutes = referenceMinutes / gpu.relativeSpeed;
  const lowMinutes = Math.max(1, estimatedMinutes * 0.7);
  const highMinutes = Math.max(lowMinutes + 1, estimatedMinutes * 1.8 + 2);
  const cpuCores = Math.max(1, Number(form.num_workers) * Number(form.threads_per_worker) || 1);
  const memory = Math.max(1, memoryGiB(form.worker_memory_limit) * Math.max(1, Number(form.num_workers) || 1));
  const combinedRate = gpu.pricePerSecond + cpuCores * 0.0000131 + memory * 0.00000222;
  const lowCost = lowMinutes * 60 * combinedRate;
  const highCost = highMinutes * 60 * combinedRate;
  const analyzedText = analysis?.image_count
    ? `${analysis.image_count} images · ${analysis.total_megapixels.toLocaleString()} source MP · ${formatBytes(analysis.image_bytes)}`
    : `${imageCount} catalog images · resolution assumed`;

  return <section className="modal-estimate" aria-label="Estimated Modal compute cost">
    <div className="modal-estimate-heading"><div><span>ESTIMATED MODAL COMPUTE</span><strong>{formatCost(lowCost)}–{formatCost(highCost)}</strong></div><em>{formatDuration(lowMinutes)}–{formatDuration(highMinutes)}</em></div>
    <div className="modal-estimate-bar"><span style={{ width: `${Math.min(100, Math.max(12, (lowMinutes / highMinutes) * 100))}%` }}/></div>
    <p>{analyzedText}</p>
    <small>{displayName(form.config_name)} · {displayName(form.splat_implementation)}{form.splat_implementation === "gsplat" ? ` · ${steps.toLocaleString()} steps` : ""} · {gpu.id}</small>
    <small>Estimate includes GPU plus approximately {cpuCores} CPU core{cpuCores === 1 ? "" : "s"} and {memory.toFixed(0)} GiB memory. Actual runtime and billing vary with scene complexity, caching, and utilization.</small>
    <a href="https://modal.com/pricing" target="_blank" rel="noreferrer">Modal pricing · rates checked Aug 13, 2026 ↗</a>
  </section>;
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
}

function Section({ number, title, subtitle, children }: SectionProps) {
  return <section className="form-section">
    <div className="section-heading"><span className="step-number">{number}</span><div><strong>{title}</strong><small>{subtitle}</small></div></div>
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
    <strong>{device.label}</strong><small>{device.details}{device.memory ? ` · ${device.memory}` : ""}</small>
    <span className={`capability ${device.supports_gaussian_splatting ? "yes" : "no"}`}>{device.supports_gaussian_splatting ? "Splat ready" : "Reconstruction"}</span>
  </div>;
}

interface RunFormProps {
  schema: ConfigurationSchema;
  hardware: HardwareCatalog | null;
  samples: SampleDataset[];
  samplesLoading: boolean;
  onStarted: (job: Job) => void;
  onTabChange: (tab: string) => void;
  remotePromptKey: number;
  schemaLoading: boolean;
  schemaError: string;
  onRetrySchema: () => void;
}

function RunForm({ schema, hardware, samples, samplesLoading, onStarted, onTabChange, remotePromptKey, schemaLoading, schemaError, onRetrySchema }: RunFormProps) {
  const [form, setForm] = useState<RunFormState>(EMPTY_FORM);
  const [inputMode, setInputMode] = useState<"upload" | "sample">("upload");
  const [formatAutomatic, setFormatAutomatic] = useState(true);
  const [formatDetection, setFormatDetection] = useState<DatasetFormatDetection | null>(null);
  const [loaderOptions, setLoaderOptions] = useState<LoaderValues>({});
  const [remote, setRemote] = useState<RemoteWorkspace | null>(null);
  const [remoteMessage, setRemoteMessage] = useState("");
  const [modalTokenMessage, setModalTokenMessage] = useState("");
  const [modalDiscovering, setModalDiscovering] = useState(false);
  const [remoteChecking, setRemoteChecking] = useState(false);
  const [modalDeploying, setModalDeploying] = useState(false);
  const [modalWorkspaceIssue, setModalWorkspaceIssue] = useState(false);
  const [modalDeployment, setModalDeployment] = useState<ModalDeployment | null>(null);
  const [modalLogExpanded, setModalLogExpanded] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [datasetFolder, setDatasetFolder] = useState<UploadedFolder | null>(null);
  const [imagesFolder, setImagesFolder] = useState<UploadedFolder | null>(null);
  const [preparedSample, setPreparedSample] = useState<PreparedSample | null>(null);
  const [sampleBusy, setSampleBusy] = useState(false);
  const [sampleMessage, setSampleMessage] = useState("");
  const modalDiscoverySequence = useRef(0);
  const automaticModalEndpoint = useRef("");
  const set = <K extends keyof RunFormState>(key: K, value: RunFormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  useEffect(() => {
    if (!hardware) return;
    const splatDevice = hardware.devices.find((item) => item.supports_gaussian_splatting);
    setForm((current) => ({
      ...current,
      hardware: splatDevice?.id ?? hardware.devices[0]?.id ?? "cpu",
      splat_implementation: current.execution_target === "local" && !splatDevice ? "none" : current.splat_implementation,
    }));
  }, [hardware]);

  useEffect(() => {
    if (remotePromptKey < 1) return;
    setForm((current) => ({
      ...current,
      execution_target: "remote",
      remote_connection: "api",
      remote_provider: "modal",
      splat_implementation: current.splat_implementation === "none" ? schema.defaults.splat_implementation : current.splat_implementation,
    }));
    window.setTimeout(() => document.getElementById("computeTarget")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }, [remotePromptKey, schema]);

  const modelCatalog = remote?.configuration?.models ?? schema.models;
  const model = modelCatalog.find((item) => item.id === form.config_name);
  const capabilities = model?.capabilities ?? { iterative_splat: false, mvs: false, share_intrinsics: false };
  const splatOptions = useMemo(() => schema.splat_implementations.map((item) => ({
    ...item, status: item.id === "gsplat" && !capabilities.iterative_splat ? "disabled" : "available",
  })), [schema, capabilities.iterative_splat]);
  const splatMeta = schema.splat_implementations.find((item) => item.id === form.splat_implementation);
  const selectedHardware = hardware?.devices.find((item) => item.id === form.hardware);
  const machineProfile = advancedMachineProfile(form, hardware, selectedHardware);
  const selectedSample = samples.find((item) => item.id === form.sample_id);
  const formatOptions: Choice[] = [{ id: "auto", label: "Auto-detect · Recommended" }, ...schema.loaders.map((loader) => ({ id: loader, label: displayName(loader) }))];
  const datasetAnalysis = inputMode === "sample"
    ? preparedSample?.analysis
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
  const modalGpuOptions = MODAL_GPU_PRICING.map((item) => ({
    ...item,
    label: item.id === modalGpuRecommendation?.gpu.id ? `${item.label} · Recommended` : item.label,
  }));
  const remoteProviders: Choice[] = [
    { id: "modal", label: "Modal" },
    { id: "lambda", label: "Lambda Cloud — Coming Soon!", status: "coming-soon" },
    { id: "runpod", label: "RunPod — Coming Soon!", status: "coming-soon" },
    { id: "vast", label: "Vast.ai — Coming Soon!", status: "coming-soon" },
    { id: "aws", label: "AWS EC2 — Coming Soon!", status: "coming-soon" },
  ];

  const useDeployedModalWorkspace = (gpu: string) => {
    const gpuInfo = MODAL_GPU_PRICING.find((item) => item.id === gpu);
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
    set("remote_hardware", "cuda:0");
    return workspace;
  };

  useEffect(() => {
    if (!modalGpuRecommendation) return;
    if (form.modal_gpu === modalGpuRecommendation.gpu.id) return;
    set("modal_gpu", modalGpuRecommendation.gpu.id);
    if (form.remote_endpoint) {
      setRemote(null);
      setModalWorkspaceIssue(true);
      setRemoteMessage("The dataset changed the recommended VM. Update the Modal workspace to apply it, then verify readiness.");
    }
  }, [modalGpuRecommendation?.gpu.id, modalRecommendationKey, form.modal_gpu, form.remote_endpoint]);

  useEffect(() => {
    setForm((current) => ({
      ...current,
      num_workers: machineProfile.workers,
      threads_per_worker: machineProfile.threadsPerWorker,
      worker_memory_limit: `${machineProfile.memoryPerWorkerGiB}GB`,
    }));
  }, [machineProfile.key]);

  useEffect(() => {
    if (!selectedSample || preparedSample?.path !== "") return;
    setSampleMessage(form.execution_target === "remote"
      ? "Will download directly on Modal"
      : selectedSample.prepared ? "Cached and ready" : "Will download when the run starts");
  }, [form.execution_target, selectedSample, preparedSample?.path]);

  const updateModalCredential = (field: "modal_token_id" | "modal_token_secret", value: string) => {
    modalDiscoverySequence.current += 1;
    setModalDiscovering(false);
    setRemoteChecking(false);
    const parsed = parseModalTokenCommand(value);
    setRemote(null);
    setModalWorkspaceIssue(false);
    setRemoteMessage("");
    if (parsed) {
      setForm((current) => ({ ...current, modal_token_id: parsed.tokenId, modal_token_secret: parsed.tokenSecret, modal_api_key: "" }));
      setModalTokenMessage("Token command parsed. Both fields are filled.");
      return;
    }
    setForm((current) => ({ ...current, [field]: value, modal_api_key: "" }));
    setModalTokenMessage("");
  };

  const updateModalGpu = (value: string) => {
    set("modal_gpu", value);
    setRemote(null);
    setModalWorkspaceIssue(Boolean(form.remote_endpoint));
    setRemoteMessage("VM selection changed. Update the Modal workspace to apply it, then verify readiness.");
  };

  useEffect(() => {
    if (form.execution_target !== "remote" || form.remote_connection !== "api" || form.remote_provider !== "modal") return;
    if (!form.modal_token_id.startsWith("ak-") || !form.modal_token_secret.startsWith("as-")) return;
    const sequence = ++modalDiscoverySequence.current;
    const timer = window.setTimeout(async () => {
      setModalDiscovering(true);
      setRemoteMessage("Finding your deployed GTSFM app on Modal…");
      try {
        const discovered = await getJson<ModalDiscovery>("/api/modal/discover", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token_id: form.modal_token_id, token_secret: form.modal_token_secret }),
        });
        if (sequence !== modalDiscoverySequence.current) return;
        setForm((current) => {
          if (current.remote_endpoint && current.remote_endpoint !== automaticModalEndpoint.current) {
            return { ...current, modal_api_key: discovered.api_key };
          }
          return { ...current, remote_endpoint: discovered.endpoint, modal_api_key: discovered.api_key };
        });
        automaticModalEndpoint.current = discovered.endpoint;
        useDeployedModalWorkspace(form.modal_gpu);
        setModalWorkspaceIssue(false);
        setRemoteMessage(`Found ${discovered.app_name} · ${discovered.function_name}. Update it to this GTSFM version, or verify the existing workspace.`);
      } catch (reason) {
        if (sequence === modalDiscoverySequence.current) {
          setModalWorkspaceIssue(true);
          setRemoteMessage(errorMessage(reason));
        }
      } finally {
        if (sequence === modalDiscoverySequence.current) setModalDiscovering(false);
      }
    }, 450);
    return () => window.clearTimeout(timer);
  }, [form.execution_target, form.remote_connection, form.remote_provider, form.modal_token_id, form.modal_token_secret]);

  useEffect(() => {
    if (form.splat_implementation === "gsplat" && model && !capabilities.iterative_splat) set("splat_implementation", "none");
  }, [form.splat_implementation, model, capabilities.iterative_splat]);

  async function inspectRemote(endpoint: string, apiKey: string) {
    const payload = await getJson<RemoteWorkspace>("/api/remote/inspect", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint, api_key: apiKey, remote_provider: form.remote_provider }) });
    setRemote({ ...payload, verified: true });
    const gsDevice = payload.hardware.devices.find((item) => item.supports_gaussian_splatting);
    set("remote_hardware", gsDevice?.id ?? payload.hardware.devices[0]?.id ?? "");
    return payload;
  }

  const connectRemote = async () => {
    if (form.remote_connection !== "api") {
      setRemoteMessage("SSH connection testing is coming soon. You can finish the VM details now.");
      return;
    }
    if (!form.remote_endpoint || !modalBearerToken(form)) {
      setRemoteMessage("Deploy the GTSFM Modal workspace first, or wait for an existing deployment to be discovered.");
      return;
    }
    setRemoteChecking(true);
    setModalWorkspaceIssue(false);
    setRemote((current) => current ? { ...current, verified: false } : null);
    setRemoteMessage("Checking the lightweight Modal control service…");
    try {
      const payload = await inspectRemote(form.remote_endpoint, modalBearerToken(form));
      setModalWorkspaceIssue(false);
      setRemoteMessage(`${payload.hardware.summary}. The control service is ready; the GPU stays off until you run a reconstruction.`);
    } catch (reason) {
      setModalWorkspaceIssue(true);
      setRemoteMessage(`Workspace check failed. Update the Modal workspace before running. ${errorMessage(reason)}`);
    } finally {
      setRemoteChecking(false);
    }
  };

  const deployModal = async () => {
    if (!form.modal_token_id.startsWith("ak-") || !form.modal_token_secret.startsWith("as-")) {
      setRemoteMessage("Enter both Modal token fields before deploying.");
      return;
    }
    modalDiscoverySequence.current += 1;
    setModalDeploying(true);
    setModalWorkspaceIssue(false);
    setRemote(null);
    setRemoteMessage("");
    try {
      let deployment = await getJson<ModalDeployment>("/api/modal/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token_id: form.modal_token_id,
          token_secret: form.modal_token_secret,
          gpu: form.modal_gpu,
          cpu: Math.max(1, Number(form.num_workers) * Number(form.threads_per_worker) || 1),
          memory_mb: Math.max(4096, Math.round(memoryGiB(form.worker_memory_limit) * Math.max(1, Number(form.num_workers) || 1) * 1024)),
        }),
      });
      setModalDeployment(deployment);
      while (["queued", "running", "cancelling"].includes(deployment.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        deployment = await getJson<ModalDeployment>(`/api/modal/deploy/${encodeURIComponent(deployment.id)}`);
        setModalDeployment(deployment);
      }
      if (deployment.status === "cancelled") {
        setRemoteMessage("Modal workspace setup stopped.");
        return;
      }
      if (deployment.status === "failed") throw new Error(deployment.error || "Modal deployment failed");
      automaticModalEndpoint.current = deployment.endpoint;
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
        await inspectRemote(deployment.endpoint, deployment.api_key);
      } catch (reason) {
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
        setRemoteChecking(false);
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
    } catch (reason) {
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
      setFormatDetection(null);
      setForm((current) => ({ ...current, sample_id: "", dataset_dir: "" }));
      return;
    }
    const recommendations = sample.recommendations;
    setFormatAutomatic(true);
    setFormatDetection({
      loader: recommendations.loader,
      confidence: 1,
      reason: "Verified from the sample's upstream GitHub directory structure.",
      loader_options: recommendations.loader_options ?? {},
      alternatives: [],
    });
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
    event.preventDefault(); setBusy(true); setError("");
    try {
      if (form.execution_target === "remote" && form.remote_connection === "ssh") {
        throw new Error("SSH execution is not available yet. Choose API to run on Modal.");
      }
      if (form.execution_target === "remote" && form.remote_provider === "modal" && !remote?.verified) {
        throw new Error("The Modal workspace must pass its health check before a reconstruction can start.");
      }
      const payload = { ...form,
        loader: formatAutomatic ? "auto" : form.loader, api_key: form.execution_target === "remote" ? modalBearerToken(form) : "", loader_options: loaderOptions,
        hardware: form.execution_target === "remote" ? form.remote_hardware : form.hardware,
        max_resolution: form.max_resolution ? Number(form.max_resolution) : null,
        num_workers: Number(form.num_workers), threads_per_worker: Number(form.threads_per_worker),
        gs_max_steps: Number(form.gs_max_steps), live_preview_interval: Number(form.live_preview_interval),
        max_frame_lookahead: form.max_frame_lookahead ? Number(form.max_frame_lookahead) : null,
        num_matched: form.num_matched ? Number(form.num_matched) : null,
        num_retry_cluster_connection: form.num_retry_cluster_connection ? Number(form.num_retry_cluster_connection) : null };
      const job = await getJson<Job>("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      onStarted(job); onTabChange("activity");
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const modalWorkspaceSelected = form.execution_target === "remote" && form.remote_connection === "api" && form.remote_provider === "modal";
  const modalWorkspaceState: ModalWorkspaceReadiness = remote?.verified
    ? "ready"
    : modalDeploying || modalDiscovering || remoteChecking
      ? "working"
      : modalWorkspaceIssue
        ? "attention"
        : form.remote_endpoint
          ? "found"
        : "idle";
  const modalWorkspaceDetail = remoteMessage || {
    idle: "Enter Modal credentials, then deploy or discover a workspace.",
    found: "Choose Update to apply this GTSFM version, or verify the existing deployment.",
    working: "Preparing the lightweight control service. The GPU starts only when a reconstruction runs.",
    ready: "Health checks passed. Reconstruction can start.",
    attention: "Update or verify this workspace before starting a run.",
  }[modalWorkspaceState];
  const modalActionLabel = modalDeploying
    ? modalDeployment?.phase === "verifying" ? "Starting & verifying Modal workspace…" : "Setting up Modal workspace…"
    : modalDiscovering
      ? "Finding Modal workspace…"
      : remoteChecking
        ? "Starting & verifying Modal workspace…"
        : modalWorkspaceSelected && !remote?.verified
          ? "Prepare Modal workspace first"
          : "Run reconstruction";

  return <form id="runForm" onSubmit={submit}>
    <div className="panel-intro"><span>NEW RECONSTRUCTION</span>
      {schemaLoading ? <small className="catalog-sync"><RefreshCw className="spin" size={10}/> Syncing workspace options…</small>
        : schemaError ? <button className="catalog-sync failed" type="button" onClick={onRetrySchema}><CircleAlert size={10}/> Options offline · Retry</button> : null}
      <p>Configure the source, pipeline, and compute target.</p>
    </div>
    <Section number="01" title="Input" subtitle="Choose the images to reconstruct">
      <TextField label="Run name" value={form.name} onChange={(value) => set("name", value)} autoComplete="off" />
      <div className="segmented input-source" role="group" aria-label="Input source">
        <button type="button" className={`target-choice ${inputMode === "upload" ? "active" : ""}`} onClick={() => {
          setInputMode("upload"); setFormatAutomatic(true); setFormatDetection(datasetFolder?.format_detection ?? null);
          setLoaderOptions(datasetFolder?.format_detection.loader_options ?? {});
          setForm((current) => ({ ...current, sample_id: "", dataset_dir: datasetFolder?.path ?? "", images_dir: imagesFolder?.path ?? "", loader: datasetFolder?.format_detection.loader ?? current.loader }));
        }}><FolderUp size={13} /> Upload your own</button>
        <button type="button" className={`target-choice ${inputMode === "sample" ? "active" : ""}`} onClick={() => {
          setInputMode("sample"); setFormatAutomatic(true);
          const recommendations = preparedSample?.sample.recommendations;
          setFormatDetection(recommendations ? { loader: recommendations.loader, confidence: 1, reason: "Verified from the sample's upstream GitHub directory structure.", loader_options: recommendations.loader_options ?? {}, alternatives: [] } : null);
          setLoaderOptions(recommendations?.loader_options ?? {});
          setForm((current) => ({ ...current, sample_id: preparedSample?.sample.id ?? "", dataset_dir: preparedSample?.path ?? "", images_dir: "", loader: recommendations?.loader ?? current.loader }));
        }}><Database size={13} /> GTSFM samples</button>
      </div>
      {inputMode === "upload" ? <>
        <FolderDropField label="Dataset folder" value={datasetFolder} onError={setError} onUploaded={(folder) => {
          setDatasetFolder(folder); set("dataset_dir", folder?.path ?? "");
          setFormatDetection(folder?.format_detection ?? null);
          if (formatAutomatic && folder?.format_detection) {
            set("loader", folder.format_detection.loader);
            setLoaderOptions(folder.format_detection.loader_options ?? {});
          }
          if (folder && form.name === "my-scene") set("name", folder.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "my-scene");
        }} />
        <FolderDropField label="Separate images folder" optional value={imagesFolder} onError={setError} onUploaded={(folder) => {
          setImagesFolder(folder); set("images_dir", folder?.path ?? "");
        }} />
      </> : <div className="sample-picker">
        <SelectField label="Sample scene" value={form.sample_id} options={samples} empty={samplesLoading ? "Loading GTSFM samples…" : "Choose a GTSFM sample…"} onChange={chooseSample} disabled={sampleBusy || samplesLoading} />
        {form.sample_id && <div className={`sample-card ${preparedSample ? "ready" : ""}`}>
          <span className="sample-state">{sampleBusy ? <RefreshCw className="spin" size={14} /> : preparedSample ? <Check size={14} /> : <Database size={14} />}</span>
          <div><strong>{samples.find((item) => item.id === form.sample_id)?.label}</strong><small>{samples.find((item) => item.id === form.sample_id)?.description}</small>
            <span>{sampleMessage} · {samples.find((item) => item.id === form.sample_id)?.image_count} images · <a href={samples.find((item) => item.id === form.sample_id)?.source_url} target="_blank" rel="noreferrer">View on GitHub ↗</a></span></div>
        </div>}
        <p className="field-help sample-help">Dataset format, VGGT model, resolution, and available loader settings are applied automatically.</p>
      </div>}
      <SelectField label="Dataset format" value={formatAutomatic ? "auto" : form.loader} options={formatOptions} onChange={(value) => {
        if (value === "auto") {
          setFormatAutomatic(true);
          if (formatDetection) {
            set("loader", formatDetection.loader);
            setLoaderOptions(formatDetection.loader_options ?? {});
          }
          return;
        }
        setFormatAutomatic(false); set("loader", value); setLoaderOptions({});
      }} />
      {formatAutomatic && <p className={`field-help format-detection ${formatDetection?.confidence === 0 ? "warning" : ""}`}>
        {formatDetection ? <><strong>{displayName(formatDetection.loader)}</strong> · {formatDetection.reason}</> : "Upload a dataset or choose a GitHub sample and the backend will inspect its directory structure."}
      </p>}
      <LoaderOptions descriptors={schema.loader_options[form.loader] ?? []} values={loaderOptions} setValues={setLoaderOptions} />
    </Section>
    <Section number="02" title="Models" subtitle="VGGT is the default reconstruction model">
      <SelectField label="Reconstruction model" value={form.config_name} options={modelCatalog} onChange={(value) => set("config_name", value)} />
      <SelectField label="Splat implementation" value={form.splat_implementation} options={splatOptions} onChange={(value) => set("splat_implementation", value)} />
      <p className="field-help">{splatMeta?.description}</p>
      {form.splat_implementation === "gsplat" && <div className="nested-options">
        <SelectField label="Optimizer preset" value={form.gaussian_splatting_config_name} options={remote?.configuration?.gaussian_splatting_models ?? schema.gaussian_splatting_models} onChange={(value) => set("gaussian_splatting_config_name", value)} />
        <div className="form-grid">
          <TextField label="Training steps" type="number" min="1" step="1" value={form.gs_max_steps} onChange={(value) => set("gs_max_steps", value)} />
          <TextField label="Preview every" type="number" min="10" step="10" value={form.live_preview_interval} onChange={(value) => set("live_preview_interval", value)} />
        </div>
      </div>}
      <Toggle id="runMvs" checked={form.run_mvs} disabled={!capabilities.mvs} onCheckedChange={(value) => set("run_mvs", value)}>Also run dense MVS</Toggle>
    </Section>
    <Section number="03" title="Compute" subtitle="Run here or on a remote VM">
      <div className="segmented" id="computeTarget" role="group" aria-label="Execution target">
        <button type="button" className={`target-choice ${form.execution_target === "local" ? "active" : ""}`} onClick={() => set("execution_target", "local")}><Cpu size={13} /> This machine</button>
        <button type="button" className={`target-choice ${form.execution_target === "remote" ? "active" : ""}`} onClick={() => setForm((current) => ({ ...current, execution_target: "remote", splat_implementation: current.splat_implementation === "none" ? schema.defaults.splat_implementation : current.splat_implementation }))}><Server size={13} /> Remote VM</button>
      </div>
      {form.execution_target === "local" ? <>
        {hardware ? <><SelectField label="Hardware" value={form.hardware} options={hardware.devices} onChange={(value) => set("hardware", value)} /><HardwareCard device={selectedHardware} /></> : <div className="hardware-card detecting"><RefreshCw className="spin" size={13}/><div><strong>Detecting compute devices…</strong><small>You can configure the rest of the run while this finishes.</small></div></div>}
      </> : <div className="nested-options remote-vm-options">
        <SelectField label="Connection method" value={form.remote_connection} options={[
          { id: "api", label: "API" }, { id: "ssh", label: "SSH" },
        ]} onChange={(value) => { set("remote_connection", value as RunFormState["remote_connection"]); setRemote(null); setRemoteMessage(""); }} />
        {form.remote_connection === "api" ? <>
          <SelectField label="Service" value={form.remote_provider} options={remoteProviders} onChange={(value) => { set("remote_provider", value); setRemote(null); }} />
          {form.remote_provider === "modal" && <div className="provider-panel">
            <div className="provider-heading"><span className="provider-mark">M</span><div><strong>Modal</strong><small>Connect with your Modal account token</small></div><span className="provider-status">AVAILABLE</span></div>
            <SelectField label="Modal VM GPU" value={form.modal_gpu} options={modalGpuOptions} onChange={updateModalGpu} />
            <ModalGpuRecommendationCard recommendation={modalGpuRecommendation} selectedGpu={form.modal_gpu} onApply={() => {
              if (modalGpuRecommendation) updateModalGpu(modalGpuRecommendation.gpu.id);
            }}/>
            <ModalCostEstimate form={form} analysis={datasetAnalysis} fallbackImageCount={selectedSample?.image_count ?? 0}/>
            <p className="field-help modal-command-help">Enter the two values separately, or paste the complete <code>modal token set --token-id … --token-secret …</code> command into either field.</p>
            <div className="form-grid">
              <TextField label="Token ID" value={form.modal_token_id} onChange={(value) => updateModalCredential("modal_token_id", value)} autoComplete="off" spellCheck={false} placeholder="ak-…" />
              <TextField label="Token secret" type="password" value={form.modal_token_secret} onChange={(value) => updateModalCredential("modal_token_secret", value)} autoComplete="new-password" spellCheck={false} placeholder="as-…" />
            </div>
            {modalTokenMessage && <div className="credential-success"><Check size={12} /> {modalTokenMessage}</div>}
            <TextField label="GTSFM endpoint" type="url" value={form.remote_endpoint} onChange={(value) => {
              set("remote_endpoint", value);
              setRemote(null);
              setModalWorkspaceIssue(false);
              setRemoteMessage("Endpoint changed. Start and verify this workspace before running.");
            }} placeholder={modalDiscovering ? "Discovering your Modal endpoint…" : "Filled after credentials are verified"} />
            <button className="modal-deploy-action full-width" type="button" onClick={deployModal} disabled={modalDiscovering || modalDeploying || !form.modal_token_id || !form.modal_token_secret}><Server size={13} /> {modalDeploying ? "Setup in progress…" : form.remote_endpoint ? "Update Modal workspace" : "Set up & deploy Modal workspace"}</button>
            {modalDeployment && <ModalDeploymentProgress deployment={modalDeployment} onCancel={stopModalDeployment} onExpand={() => setModalLogExpanded(true)}/>} 
            {modalDeployment && modalLogExpanded && <ModalDeploymentDialog deployment={modalDeployment} onCancel={stopModalDeployment} onClose={() => setModalLogExpanded(false)}/>} 
            {form.remote_endpoint && form.modal_api_key && <button className="secondary-action full-width" type="button" onClick={() => connectRemote()} disabled={modalDiscovering || modalDeploying || remoteChecking}>{remoteChecking ? <RefreshCw className="spin" size={13}/> : remote?.verified ? <Check size={13}/> : <MonitorCog size={13}/>} {remoteChecking ? "Starting & checking workspace…" : remote?.verified ? "Modal workspace ready" : "Start & verify workspace"}</button>}
            <ModalWorkspaceStatus state={modalWorkspaceState} detail={modalWorkspaceDetail}/>
            {remote && <SelectField label="Remote hardware" value={form.remote_hardware} options={remote.hardware.devices} onChange={(value) => set("remote_hardware", value)} />}
          </div>}
        </> : <div className="provider-panel">
          <div className="provider-heading"><span className="provider-mark ssh"><Terminal size={14} /></span><div><strong>Direct VM</strong><small>Connect to a machine you control</small></div><span className="provider-status soon">COMING SOON!</span></div>
          <div className="form-grid">
            <TextField label="Host" value={form.ssh_host} onChange={(value) => set("ssh_host", value)} placeholder="gpu-box.example.com" />
            <TextField label="Port" type="number" min="1" max="65535" value={form.ssh_port} onChange={(value) => set("ssh_port", value)} />
            <TextField label="Username" value={form.ssh_username} onChange={(value) => set("ssh_username", value)} autoComplete="username" placeholder="ubuntu" />
            <SelectField label="Authentication" value={form.ssh_authentication} options={[
              { id: "agent", label: "SSH agent" }, { id: "key", label: "Private key" },
            ]} onChange={(value) => set("ssh_authentication", value as RunFormState["ssh_authentication"])} />
          </div>
          {form.ssh_authentication === "key" && <TextField label="Private key path" value={form.ssh_private_key} onChange={(value) => set("ssh_private_key", value)} placeholder="/Users/you/.ssh/id_ed25519" />}
          <TextField label="Remote workspace" value={form.ssh_workspace} onChange={(value) => set("ssh_workspace", value)} placeholder="~/gtsfm-workspace" />
          <p className="field-help">SSH setup is visible now; remote execution and file transfer are the next connector step.</p>
        </div>}
      </div>}
    </Section>
    <Collapsible.Root className="advanced-options" open={advanced} onOpenChange={setAdvanced}>
      <Collapsible.Trigger className="advanced-trigger"><span><SlidersHorizontal size={13} /><span className="advanced-trigger-copy">Advanced settings<small>{machineProfile.label}</small></span></span><ChevronDown size={14} /></Collapsible.Trigger>
      <Collapsible.Content className="advanced-content">
        <div className="machine-profile-summary">
          <div><span>MACHINE PROFILE</span><strong>{machineProfile.label}</strong></div>
          <p>{machineProfile.description}</p>
          <div className="machine-profile-specs"><span>{machineProfile.workers} worker{machineProfile.workers === 1 ? "" : "s"}</span><span>{machineProfile.threadsPerWorker} thread{machineProfile.threadsPerWorker === 1 ? "" : "s"} / worker</span><span>{machineProfile.memoryPerWorkerGiB} GB / worker</span></div>
        </div>
        <div className="form-grid">
          <TextField label="Max resolution" type="number" min="1" value={form.max_resolution} onChange={(value) => set("max_resolution", value)} placeholder="Model default" />
          <TextField label="Workers" type="number" min="1" value={form.num_workers} onChange={(value) => set("num_workers", value)} disabled={!machineProfile.allowWorkers} title={!machineProfile.allowWorkers ? "Fixed for the selected single-GPU machine" : undefined} />
          <TextField label="Threads / worker" type="number" min="1" value={form.threads_per_worker} onChange={(value) => set("threads_per_worker", value)} disabled={!machineProfile.allowThreads} />
          <TextField label="Memory / worker" value={form.worker_memory_limit} onChange={(value) => set("worker_memory_limit", value)} disabled={!machineProfile.allowMemory} />
        </div>
        {!machineProfile.allowWorkers && <p className="advanced-machine-note">Worker count is locked because the selected machine exposes one GPU. Choose a CPU machine to distribute work across multiple workers.</p>}
        <SelectField label="Graph partitioner" value={form.graph_partitioner} options={schema.graph_partitioners} empty="Model default" onChange={(value) => set("graph_partitioner", value)} />
        <div className="form-grid">
          <SelectField label="Global descriptor" value={form.global_descriptor_config_name} options={schema.global_descriptors} empty="Model default" onChange={(value) => set("global_descriptor_config_name", value)} />
          <SelectField label="Image retriever" value={form.retriever_config_name} options={schema.retrievers} empty="Model default" onChange={(value) => set("retriever_config_name", value)} />
          <SelectField label="Correspondence" value={form.correspondence_generator_config_name} options={schema.correspondence_generators} empty="Model default" onChange={(value) => set("correspondence_generator_config_name", value)} />
          <SelectField label="Verifier" value={form.verifier_config_name} options={schema.verifiers} empty="Model default" onChange={(value) => set("verifier_config_name", value)} />
          <TextField label="Frame lookahead" type="number" min="0" value={form.max_frame_lookahead} onChange={(value) => set("max_frame_lookahead", value)} placeholder="Model default" />
          <TextField label="Matches / image" type="number" min="0" value={form.num_matched} onChange={(value) => set("num_matched", value)} placeholder="Model default" />
        </div>
        <Toggle id="shareIntrinsics" checked={form.share_intrinsics} disabled={!capabilities.share_intrinsics} onCheckedChange={(value) => set("share_intrinsics", value)}>Share camera intrinsics</Toggle>
        <div className="form-grid">
          <SelectField label="Log level" value={form.log} options={schema.log_levels} onChange={(value) => set("log", value)} />
          <TextField label="Dashboard port" value={form.dashboard_port} onChange={(value) => set("dashboard_port", value)} placeholder=":8787" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Input worker" value={form.input_worker} onChange={(value) => set("input_worker", value)} placeholder="Optional worker address" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Dask temp folder" value={form.dask_tmpdir} onChange={(value) => set("dask_tmpdir", value)} placeholder="System default" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Cluster config" value={form.cluster_config} onChange={(value) => set("cluster_config", value)} placeholder="Optional YAML path" disabled={!machineProfile.allowLocalRuntime} />
          <TextField label="Cluster retries" type="number" min="0" value={form.num_retry_cluster_connection} onChange={(value) => set("num_retry_cluster_connection", value)} placeholder="3" disabled={!machineProfile.allowLocalRuntime} />
        </div>
        <Field label="Hydra overrides"><textarea rows={4} value={form.advanced_overrides} onChange={(event) => set("advanced_overrides", event.target.value)} /></Field>
      </Collapsible.Content>
    </Collapsible.Root>
    <div className="form-error" role="alert">{error}</div>
    <button className="primary-action" type="submit" disabled={busy || sampleBusy || modalDeploying || modalDiscovering || remoteChecking || (inputMode === "sample" && !preparedSample) || (form.execution_target === "remote" && (!form.remote_endpoint || !form.modal_api_key || !remote?.verified))}><span>{busy ? "Starting…" : sampleBusy ? "Preparing sample…" : modalActionLabel}</span>{busy || sampleBusy || modalDeploying || modalDiscovering || remoteChecking ? <RefreshCw className="spin" size={14} /> : <Play size={14} fill="currentColor" />}</button>
  </form>;
}

const statusLabel = (status: JobStatus): string => ({ queued: "Queued", running: "Running", completed: "Completed", failed: "Failed", cancelled: "Cancelled" })[status];

function SplatDownload({ jobId, compact = false }: { jobId: string; compact?: boolean }) {
  const href = `/api/jobs/${encodeURIComponent(jobId)}/splat?format=ply`;
  return <div className={`splat-download ${compact ? "compact" : ""}`}>
    <span className="splat-format-label">.PLY</span>
    <a href={href} download title="Download PLY splat"><Download size={12}/>{!compact && <span>Download</span>}</a>
  </div>;
}

interface ActivityPanelProps {
  jobs: Job[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCancel: (id: string) => void;
  onRefresh: () => void;
}

function ActivityPanel({ jobs, activeId, onSelect, onCancel, onRefresh }: ActivityPanelProps) {
  return <div><div className="panel-title"><div><h3>Activity</h3><p>Running and recent jobs</p></div><button className="icon-button" title="Refresh" onClick={onRefresh}><RefreshCw size={14} /></button></div>
    <div className="job-list">{jobs.length ? jobs.map((job) => <article key={job.id} className={`job-card ${job.id === activeId ? "selected" : ""}`}>
      <div className="job-card-top"><strong>{job.name}</strong><span className={`job-status ${job.status}`}>{statusLabel(job.status)}</span></div>
      <small>{displayName(job.spec.config_name || "GTSFM")} · {displayName(job.spec.splat_implementation || "no_splats")}</small>
      {job.error && <p className="job-error">{job.error}</p>}
      <div className="job-actions"><button className="text-button" onClick={() => onSelect(job.id)}>{job.has_final_splat ? "View splat" : ["queued", "running"].includes(job.status) ? "View progress" : "View details"}</button>
        {["queued", "running"].includes(job.status) && <button className="text-button danger" onClick={() => onCancel(job.id)}><Square size={9} fill="currentColor"/> Stop</button>}
        {job.status === "completed" && !job.remote && <a className="text-button" href="/?view=results">View results</a>}
        {job.remote?.workspace_url && <a className="text-button" href={job.remote.workspace_url} target="_blank" rel="noreferrer">Remote results ↗</a>}
      </div>{job.has_final_splat && <SplatDownload jobId={job.id} compact />}</article>) : <div className="empty-state"><Activity size={20} /><strong>No runs yet</strong><span>Configure your first reconstruction in New run.</span></div>}</div>
  </div>;
}

function ResultsPanel() {
  return <div><div className="panel-title"><div><h3>Reconstructions</h3><p>Open a scene in the 3D viewer</p></div></div>
    <div id="info" className="info-summary" role="status"><div className="info-pill" data-role="count">—</div><div className="info-details"><div className="info-title" data-role="title">Loading reconstructions…</div><div className="info-path" data-role="path">Checking workspace…</div></div></div>
    <input type="text" id="filter" placeholder="Filter scenes…" aria-label="Filter scenes" /><div id="sceneList" />
  </div>;
}

interface StatusBarProps {
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

function StatusBar({ job, live, logsOpen, setLogsOpen, visible, onCancel, onClose }: StatusBarProps) {
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
  const style = position ? { left: position.left, top: position.top, width: position.width, right: "auto", bottom: "auto" } : undefined;
  return <div ref={barRef} className={`run-status-bar ${dragging ? "is-dragging" : ""}`} data-status={job.status} style={style} title="Drag to move run status" onPointerDown={beginDrag} onPointerMove={moveDrag} onPointerUp={finishDrag} onPointerCancel={finishDrag}>
    <div className="status-copy status-drag-handle" onDoubleClick={() => setPosition(null)}><span className="status-dot"/><div><strong>{job.name} · {statusLabel(job.status)}</strong><small>{job.error || (live?.stage === "gaussian_splatting" ? "Optimizing Gaussian splats" : "Running reconstruction pipeline")}</small></div></div>
    <div className="status-metrics"><span>{live?.max_steps ? `${Number(live.step).toLocaleString()} / ${Number(live.max_steps).toLocaleString()} steps` : ""}</span><span>{loss}</span><span>{splatCount}</span></div>
    <div className="status-actions">{job.has_final_splat && <SplatDownload jobId={job.id} />}{["queued", "running"].includes(job.status) && <button className="secondary-action stop-process" type="button" onClick={() => onCancel(job.id)}><Square size={10} fill="currentColor"/> Stop</button>}<button className="secondary-action" onClick={() => setLogsOpen(!logsOpen)}><Terminal size={13}/> Logs</button><button className="status-close" type="button" title="Close run status" aria-label="Close run status" onClick={onClose}><X size={15}/></button></div>
    <Progress.Root className="run-progress-track" value={progress * 100}><Progress.Indicator className="run-progress-fill" style={{ transform: `translateX(-${100 - progress * 100}%)` }} /></Progress.Root>
  </div>;
}

interface ViewerProps {
  activeJob: Job | null;
  live: LiveState | null;
  logsOpen: boolean;
  setLogsOpen: Dispatch<SetStateAction<boolean>>;
  statusBarOpen: boolean;
  setStatusBarOpen: Dispatch<SetStateAction<boolean>>;
  onCancelJob: (id: string) => void;
  setup: SetupStatus | null;
  setupRefreshing: boolean;
  onRefreshSetup: () => void;
  onSetupChange: (setup: SetupStatus) => void;
}

interface LogPanelGeometry {
  left: number;
  top: number;
  width: number;
  height: number;
}

type LogResizeCorner = "nw" | "ne" | "sw" | "se";

function LogPanel({ open, lines, onClose }: { open: boolean; lines: string[]; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const copyReset = useRef<number | null>(null);
  const [geometry, setGeometry] = useState<LogPanelGeometry | null>(null);
  const [copied, setCopied] = useState(false);

  const copyLogs = async () => {
    if (!await copyTextToClipboard(lines.join("\n"))) return;
    setCopied(true);
    if (copyReset.current !== null) window.clearTimeout(copyReset.current);
    copyReset.current = window.setTimeout(() => setCopied(false), 1000);
  };

  useEffect(() => () => { if (copyReset.current !== null) window.clearTimeout(copyReset.current); }, []);

  const beginDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest("button")) return;
    const panel = panelRef.current;
    const container = panel?.parentElement;
    if (!panel || !container) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const panelRect = panel.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const start = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: panelRect.left - containerRect.left,
      top: panelRect.top - containerRect.top,
      width: panelRect.width,
      height: panelRect.height,
      maxLeft: containerRect.width - panelRect.width,
      maxTop: containerRect.height - panelRect.height,
    };
    setGeometry({ left: start.left, top: start.top, width: start.width, height: start.height });

    const move = (moveEvent: PointerEvent) => setGeometry({
      left: Math.max(0, Math.min(start.maxLeft, start.left + moveEvent.clientX - start.pointerX)),
      top: Math.max(0, Math.min(start.maxTop, start.top + moveEvent.clientY - start.pointerY)),
      width: start.width,
      height: start.height,
    });
    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, { once: true });
    window.addEventListener("pointercancel", finish, { once: true });
  };

  const beginResize = (corner: LogResizeCorner) => (event: React.PointerEvent<HTMLButtonElement>) => {
    const panel = panelRef.current;
    const container = panel?.parentElement;
    if (!panel || !container) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const panelRect = panel.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const start = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: panelRect.left - containerRect.left,
      top: panelRect.top - containerRect.top,
      right: panelRect.right - containerRect.left,
      bottom: panelRect.bottom - containerRect.top,
      containerWidth: containerRect.width,
      containerHeight: containerRect.height,
    };
    setGeometry({ left: start.left, top: start.top, width: panelRect.width, height: panelRect.height });

    const move = (moveEvent: PointerEvent) => {
      const dx = moveEvent.clientX - start.pointerX;
      const dy = moveEvent.clientY - start.pointerY;
      const left = corner.includes("w")
        ? Math.max(0, Math.min(start.right - 300, start.left + dx))
        : start.left;
      const right = corner.includes("e")
        ? Math.min(start.containerWidth, Math.max(start.left + 300, start.right + dx))
        : start.right;
      const top = corner.includes("n")
        ? Math.max(0, Math.min(start.bottom - 150, start.top + dy))
        : start.top;
      const bottom = corner.includes("s")
        ? Math.min(start.containerHeight, Math.max(start.top + 150, start.bottom + dy))
        : start.bottom;
      setGeometry({ left, top, width: right - left, height: bottom - top });
    };
    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, { once: true });
    window.addEventListener("pointercancel", finish, { once: true });
  };

  if (!open) return null;
  const logFontSize = geometry
    ? Math.max(8, Math.min(16, 9 + (geometry.width - 420) / 140 + (geometry.height - 240) / 120))
    : 9;
  const style = geometry ? ({
    left: geometry.left,
    top: geometry.top,
    width: geometry.width,
    height: geometry.height,
    right: "auto",
    bottom: "auto",
    "--log-font-size": `${logFontSize}px`,
  } as React.CSSProperties) : undefined;
  return <div ref={panelRef} className="log-drawer" role="dialog" aria-label="Run logs" style={style}>
    {(["nw", "ne", "sw", "se"] as LogResizeCorner[]).map((corner) => <button key={corner} className={`log-resize-handle ${corner}`} type="button" title="Resize logs" aria-label={`Resize logs from ${corner}`} onPointerDown={beginResize(corner)} />)}
    <div className="log-header" onPointerDown={beginDrag}><strong><Terminal size={12}/> Run logs</strong><div className="log-header-actions"><button className="log-copy" type="button" title="Copy all logs" aria-label="Copy all logs" onClick={copyLogs}>{copied ? <Check size={12}/> : <Copy size={12}/>}<span>{copied ? "Copied!" : "Copy"}</span></button><button type="button" title="Minimize logs" aria-label="Minimize logs" onClick={onClose}><Minus size={15}/></button><button type="button" title="Close logs" aria-label="Close logs" onClick={onClose}><X size={14}/></button></div></div>
    <pre id="runLogs">{lines.join("\n")}</pre>
  </div>;
}

function SetupPanel({ setup, refreshing, onRefresh, onSetupChange, hasActiveJob }: {
  setup: SetupStatus | null;
  refreshing: boolean;
  onRefresh: () => void;
  onSetupChange: (setup: SetupStatus) => void;
  hasActiveJob: boolean;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [closed, setClosed] = useState(false);
  const [installingId, setInstallingId] = useState<string | null>(null);
  const [actionErrors, setActionErrors] = useState<Record<string, string>>({});
  const positionClass = hasActiveJob ? "with-active-job" : "";

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

  if (closed) return <button className={`setup-reopen ${positionClass}`} type="button" title="Show setup checks" aria-label="Show setup checks" onClick={() => setClosed(false)}><Wrench size={15} /></button>;

  const status = setup?.status ?? "warning";
  const optionalUnavailable = setup?.counts.optional ?? 0;
  return <section className={`setup-panel ${collapsed ? "collapsed" : ""} ${positionClass}`} aria-label="Setup checks" aria-live="polite">
    <header className="setup-panel-header">
      <span className={`setup-overall-icon ${status}`} aria-hidden="true">{status === "ready" ? <CheckCircle2 size={16} /> : <CircleAlert size={16} />}</span>
      <div className="setup-panel-copy"><span>SETUP CHECKS</span><strong>{setup?.summary ?? "Checking environment…"}</strong></div>
      <div className="setup-panel-actions">
        <button type="button" title="Refresh checks" aria-label="Refresh setup checks" onClick={onRefresh} disabled={refreshing}><RefreshCw className={refreshing ? "spin" : ""} size={13} /></button>
        <button type="button" title={collapsed ? "Expand checks" : "Collapse checks"} aria-label={collapsed ? "Expand setup checks" : "Collapse setup checks"} onClick={() => setCollapsed((current) => !current)}>{collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}</button>
        <button type="button" title="Close checks" aria-label="Close setup checks" onClick={() => setClosed(true)}><X size={14} /></button>
      </div>
    </header>
    {!collapsed && <div className="setup-panel-body">
      {!setup ? <div className="setup-loading"><RefreshCw className="spin" size={13} /> Inspecting this machine…</div> : <>
        <div className="setup-summary-line"><span>{setup.counts.ready} passed</span>{optionalUnavailable > 0 && <span>{optionalUnavailable} optional unavailable</span>}</div>
        <ul>{setup.items.map((item) => <li key={item.id} data-state={item.state}>
          <span className="setup-check-mark" aria-hidden="true">{item.state === "ready" ? <Check size={12} /> : item.state === "error" ? <X size={12} /> : <span />}</span>
          <div className="setup-check-content"><div className="setup-check-title"><strong>{item.label}</strong><span className="setup-check-tools">{!item.required && <em>optional</em>}{item.action && <button className="setup-install" type="button" disabled={!item.action.enabled || Boolean(installingId)} title={item.action.reason || item.action.label} onClick={() => runSetupAction(item)}>{installingId === item.id ? <RefreshCw className="spin" size={9}/> : item.action.enabled ? <Download size={9}/> : null}<span>{installingId === item.id ? "Working…" : item.action.label}</span></button>}</span></div><small className={actionErrors[item.id] ? "setup-action-error" : undefined}>{actionErrors[item.id] || item.detail}</small></div>
        </li>)}</ul>
      </>}
    </div>}
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

function Viewer({ activeJob, live, logsOpen, setLogsOpen, statusBarOpen, setStatusBarOpen, onCancelJob, setup, setupRefreshing, onRefreshSetup, onSetupChange }: ViewerProps) {
  const closeStatus = () => { setStatusBarOpen(false); setLogsOpen(false); };
  const showDaskStats = Boolean(activeJob && ["queued", "running"].includes(activeJob.status));
  const expectsSplats = Boolean(activeJob && (activeJob.spec.splat_implementation || "none") !== "none");
  const visualizationAvailable = Boolean(live?.preview_url || live?.final_url || activeJob?.has_final_splat);
  const showPipelineWait = Boolean(activeJob && expectsSplats && ["queued", "running"].includes(activeJob.status) && !visualizationAvailable);
  return <main id="main-content"><StatusBar job={activeJob} live={live} logsOpen={logsOpen} setLogsOpen={setLogsOpen} visible={statusBarOpen} onCancel={onCancelJob} onClose={closeStatus} />
    <SetupPanel setup={setup} refreshing={setupRefreshing} onRefresh={onRefreshSetup} onSetupChange={onSetupChange} hasActiveJob={Boolean(activeJob && statusBarOpen)} />
    <div id="sceneStats" className={showDaskStats ? "dask-active" : undefined}><div className="stat-group" data-mode="scene"><div className="stat-pair"><span className="label">Cameras</span><span className="value" id="statCameras">0</span></div><div className="stat-pair"><span className="label">Points</span><span className="value" id="statPoints">0</span></div><div className="stat-wide"><span className="label">Image</span><span className="value" id="statImageName">—</span></div></div><div className="stat-group" data-mode="splat"><div className="stat-pair"><span className="label">Splats</span><span className="value" id="statSplats">0</span></div></div>{showDaskStats && activeJob && <DaskStats job={activeJob} live={live}/>}</div>
    <canvas id="renderCanvas"/>
    <div id="hud"><label className="background-control">BG <select id="backgroundSelect" defaultValue="dark" aria-label="Viewer background"><option value="dark">Dark</option><option value="graphite">Graphite</option><option value="light-gray">Light gray</option><option value="white">White</option></select></label><button id="prevCamBtn" className="hud-scene-only" title="Previous camera"><ChevronLeft size={14}/></button><button id="nextCamBtn" className="hud-scene-only" title="Next camera"><ChevronRight size={14}/></button><button id="toggleStats">Hide stats</button><label className="hud-scene-only"><input type="checkbox" id="toggleCams" defaultChecked/> Cameras</label><label className="hud-scene-only">Point size <input type="range" id="ptSize" min="1" max="10" defaultValue="2"/></label><button id="toggleGround" type="button" aria-pressed="true">Hide plane</button><label className="plane-height-control">Plane Y <input type="range" id="groundY" min="-5" max="5" step="0.1" defaultValue="0" aria-label="Plane vertical position"/><output id="groundYValue" htmlFor="groundY">0.0</output></label></div>
    <a className="viewport-github" href="https://github.com/borglab/gtsfm" target="_blank" rel="noreferrer" title="Open GTSFM on GitHub" aria-label="Open GTSFM GitHub repository"><Github size={16}/></a>
    <LogPanel open={logsOpen} lines={activeJob?.log_tail || []} onClose={() => setLogsOpen(false)} />
    {showPipelineWait && activeJob && <div className="pipeline-wait-overlay" role="status" aria-live="polite"><div className="pipeline-wait-content"><span>RECONSTRUCTION IN PROGRESS</span><strong>{pipelineStatusMessage(activeJob, live)}</strong><div className="pipeline-wait-dots" aria-hidden="true"><i/><i/><i/></div><small>The first live Gaussian preview will appear here automatically.</small></div></div>}
    <div id="loadingOverlay" className="loading-overlay" role="status"><div className="loading-box"><span id="loadingMessage">Loading…</span><div className="loading-progress-track"><div className="loading-progress-fill" id="loadingProgress"/></div></div></div>
  </main>;
}

function hardwareWarningContent(hardware: HardwareCatalog): { title: string; description: string } {
  const available = hardware.devices.filter((device) => !device.status || device.status === "available");
  const apple = hardware.devices.find((device) => device.kind === "mps");
  if (apple) return {
    title: "Apple Metal (MPS) is not supported",
    description: "Apple Silicon and MPS can run reconstruction, but GTSFM Gaussian splatting does not currently support this backend. Use a Remote VM with an NVIDIA GPU to generate splats.",
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

function HardwareWarning({ hardware, onClose, onUseRemote }: { hardware: HardwareCatalog; onClose: () => void; onUseRemote: () => void }) {
  const content = hardwareWarningContent(hardware);
  return <section className="hardware-warning" role="alertdialog" aria-labelledby="hardware-warning-title" aria-describedby="hardware-warning-description">
    <div className="hardware-warning-icon" aria-hidden="true"><CircleAlert size={18}/></div>
    <div className="hardware-warning-copy">
      <span>HARDWARE NOTICE</span>
      <strong id="hardware-warning-title">{content.title}</strong>
      <p id="hardware-warning-description">{content.description}</p>
      <div className="hardware-warning-actions"><button type="button" className="warning-primary" onClick={onUseRemote}><Server size={12}/> Use Remote VM</button><button type="button" onClick={onClose}>Continue without splats</button></div>
    </div>
    <button className="hardware-warning-close" type="button" title="Dismiss hardware warning" aria-label="Dismiss hardware warning" onClick={onClose}><X size={14}/></button>
  </section>;
}

const SIDEBAR_WIDTH_KEY = "gtsfm-studio-sidebar-width";
const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 760;

function constrainSidebarWidth(width: number): number {
  const viewerRoom = Math.max(SIDEBAR_MIN_WIDTH, window.innerWidth - 360);
  return Math.round(Math.min(SIDEBAR_MAX_WIDTH, viewerRoom, Math.max(SIDEBAR_MIN_WIDTH, width)));
}

function storedSidebarWidth(): number {
  try {
    const saved = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return constrainSidebarWidth(Number.isFinite(saved) && saved > 0 ? saved : 420);
  } catch {
    return 420;
  }
}

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(storedSidebarWidth);
  const [sidebarResizing, setSidebarResizing] = useState(false);
  const sidebarResizeStart = useRef<{ pointerX: number; width: number } | null>(null);
  const [tab, setTab] = useState(new URLSearchParams(location.search).get("view") === "results" ? "results" : "run");
  const [schema, setSchema] = useState<ConfigurationSchema>(BOOTSTRAP_SCHEMA);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [schemaError, setSchemaError] = useState("");
  const [hardware, setHardware] = useState<HardwareCatalog | null>(null);
  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [setupRefreshing, setSetupRefreshing] = useState(false);
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [live, setLive] = useState<LiveState | null>(null);
  const [logsOpen, setLogsOpen] = useState(false);
  const [statusBarOpen, setStatusBarOpen] = useState(true);
  const [hardwareWarningDismissed, setHardwareWarningDismissed] = useState(false);
  const [remotePromptKey, setRemotePromptKey] = useState(0);
  const [viewerLoadRequest, setViewerLoadRequest] = useState(0);
  const previewVersion = useRef<string | number | null>(null);
  const finalLoaded = useRef<string | null>(null);

  const finishSidebarResize = useCallback(() => {
    sidebarResizeStart.current = null;
    setSidebarResizing(false);
    document.body.classList.remove("sidebar-is-resizing");
  }, []);

  const startSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    if (sidebarCollapsed || event.button !== 0) return;
    event.preventDefault();
    sidebarResizeStart.current = { pointerX: event.clientX, width: sidebarWidth };
    event.currentTarget.setPointerCapture(event.pointerId);
    setSidebarResizing(true);
    document.body.classList.add("sidebar-is-resizing");
  };

  const moveSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = sidebarResizeStart.current;
    if (!start) return;
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
    try { window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); } catch { /* storage is optional */ }
  }, [sidebarWidth]);
  useEffect(() => {
    const keepSidebarOnScreen = () => setSidebarWidth((current) => constrainSidebarWidth(current));
    window.addEventListener("resize", keepSidebarOnScreen);
    return () => window.removeEventListener("resize", keepSidebarOnScreen);
  }, []);

  const refreshJobs = useCallback(async () => {
    try { const payload = await getJson<JobsResponse>("/api/jobs"); setJobs(payload.items || []); setActiveId((current) => current || payload.items?.find((job) => ["queued", "running"].includes(job.status))?.id || null); } catch (reason) { console.warn("Unable to refresh jobs", reason); }
  }, []);

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
  const activeJob = jobs.find((job) => job.id === activeId) ?? null;

  useEffect(() => { if (activeId) setStatusBarOpen(true); }, [activeId]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | null = null;
    let stopped = false;
    const connect = () => {
      socket = new WebSocket(websocketUrl("/api/events/jobs"));
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data) as JobsResponse;
        setJobs(payload.items || []);
        setActiveId((current) => current || payload.items?.find((job) => ["queued", "running"].includes(job.status))?.id || null);
      };
      socket.onclose = () => { if (!stopped) retry = window.setTimeout(connect, 1000); };
    };
    connect();
    return () => { stopped = true; if (retry !== null) clearTimeout(retry); socket?.close(); };
  }, []);

  useEffect(() => {
    if (!activeJob) { setLive(null); return; }
    setLive(null);
    const socket = new WebSocket(websocketUrl(`/api/events/jobs/${encodeURIComponent(activeJob.id)}`));
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as JobEvent;
      setJobs((current) => current.map((job) => job.id === payload.job.id ? payload.job : job));
      setLive(payload.live);
    };
    return () => socket.close();
  }, [activeJob?.id]);

  useEffect(() => {
    if (!activeJob || !live) return;
    const finalKey = activeJob.status === "completed" && live.final_url ? `${activeJob.id}:${live.final_url}` : null;
    const previewKey = live.preview_url ? `${activeJob.id}:${String(live.preview_version ?? live.preview_url)}` : null;
    const desired = finalKey
      ? { kind: "final" as const, key: finalKey, url: live.final_url as string, label: `${activeJob.name} · final` }
      : previewKey
        ? { kind: "preview" as const, key: previewKey, url: live.preview_url as string, label: `${activeJob.name} · live` }
        : null;
    if (!desired) return;
    if (desired.kind === "final" && finalLoaded.current === desired.key) return;
    if (desired.kind === "preview" && previewVersion.current === desired.key) return;

    let cancelled = false;
    let retry: number | null = null;
    let loadFailures = 0;
    const tryLoad = async () => {
      if (cancelled) return;
      const viewer = window.gtsfmViewer;
      if (!viewer || viewer.isBusy()) {
        retry = window.setTimeout(() => { void tryLoad(); }, 250);
        return;
      }
      const loaded = await viewer.loadSplatsFile({ splatsUrl: desired.url, label: desired.label });
      if (cancelled) return;
      if (loaded === false) {
        loadFailures += 1;
        if (loadFailures < 3) retry = window.setTimeout(() => { void tryLoad(); }, 1200);
        return;
      }
      if (desired.kind === "final") finalLoaded.current = desired.key;
      else previewVersion.current = desired.key;
    };
    void tryLoad();
    return () => {
      cancelled = true;
      if (retry !== null) window.clearTimeout(retry);
    };
  }, [activeJob?.id, activeJob?.status, live?.final_url, live?.preview_url, live?.preview_version, viewerLoadRequest]);

  const cancelJob = async (id: string) => { await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); await refreshJobs(); };
  const activeCount = jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  const hasNvidiaSplatSupport = hardware?.devices.some((device) => device.supports_gaussian_splatting && (!device.status || device.status === "available")) ?? false;
  const showHardwareWarning = Boolean(hardware && !hasNvidiaSplatSupport && !hardwareWarningDismissed);
  const useRemoteVm = () => { setHardwareWarningDismissed(true); setTab("run"); setRemotePromptKey((current) => current + 1); };
  return <div className="app-shell">{showHardwareWarning && hardware && <HardwareWarning hardware={hardware} onClose={() => setHardwareWarningDismissed(true)} onUseRemote={useRemoteVm}/>}<aside id="sidebar" className={`${sidebarCollapsed ? "sidebar-collapsed" : ""} ${sidebarResizing ? "sidebar-resizing" : ""}`} style={{ "--sidebar-width": `${sidebarWidth}px` } as React.CSSProperties}><Brand collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((current) => !current)} />
    <Tabs.Root className="workspace-tabs" value={tab} onValueChange={setTab}>
      <Tabs.List className="studio-tabs" aria-label="Workspace sections"><Tabs.Trigger className="studio-tab" value="run"><Play size={12}/> New run</Tabs.Trigger><Tabs.Trigger className="studio-tab" value="activity"><Activity size={12}/> Activity {activeCount > 0 && <span id="activeJobCount">{activeCount}</span>}</Tabs.Trigger><Tabs.Trigger className="studio-tab" value="results"><Box size={12}/> Results</Tabs.Trigger></Tabs.List>
      <Tabs.Content className="studio-panel" value="run" forceMount><RunForm schema={schema} hardware={hardware} samples={samples} samplesLoading={samplesLoading} onStarted={(job) => { setStatusBarOpen(true); setActiveId(job.id); refreshJobs(); }} onTabChange={setTab} remotePromptKey={remotePromptKey} schemaLoading={schemaLoading} schemaError={schemaError} onRetrySchema={loadSchema}/></Tabs.Content>
      <Tabs.Content className="studio-panel" value="activity" forceMount><ActivityPanel jobs={jobs} activeId={activeId} onSelect={(id) => { setStatusBarOpen(true); setActiveId(id); previewVersion.current = null; finalLoaded.current = null; setViewerLoadRequest((current) => current + 1); }} onCancel={cancelJob} onRefresh={refreshJobs}/></Tabs.Content>
      <Tabs.Content className="studio-panel" value="results" forceMount><ResultsPanel /></Tabs.Content>
    </Tabs.Root><div className="sidebar-resize-handle" role="separator" aria-label="Resize side panel" aria-orientation="vertical" aria-valuemin={SIDEBAR_MIN_WIDTH} aria-valuemax={SIDEBAR_MAX_WIDTH} aria-valuenow={sidebarWidth} tabIndex={sidebarCollapsed ? -1 : 0} title="Drag to resize side panel" onPointerDown={startSidebarResize} onPointerMove={moveSidebarResize} onPointerUp={finishSidebarResize} onPointerCancel={finishSidebarResize} onDoubleClick={() => setSidebarWidth(constrainSidebarWidth(420))} onKeyDown={resizeSidebarWithKeyboard}/></aside><Viewer activeJob={activeJob} live={live} logsOpen={logsOpen} setLogsOpen={setLogsOpen} statusBarOpen={statusBarOpen} setStatusBarOpen={setStatusBarOpen} onCancelJob={cancelJob} setup={setup} setupRefreshing={setupRefreshing} onRefreshSetup={refreshSetup} onSetupChange={setSetup}/></div>;
}

const root = document.getElementById("root");
if (!root) throw new Error("GTSFM Studio root element is missing");
flushSync(() => createRoot(root).render(<App />));
