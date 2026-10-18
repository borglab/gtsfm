import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const vendorDirectory = fileURLToPath(new URL("../../static/vendor", import.meta.url));
const babylonSource = fileURLToPath(new URL("../node_modules/babylonjs/babylon.js", import.meta.url));
const babylonTarget = fileURLToPath(new URL("../../static/vendor/babylon.js", import.meta.url));

await mkdir(vendorDirectory, { recursive: true });
await copyFile(babylonSource, babylonTarget);
