/** Browser regressions run against the production bundle, with isolated API fixtures. */
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { chromium } = require('../../visualization/frontend/node_modules/playwright');

const root = path.resolve(__dirname, '../../visualization');
const config = {
  defaults: { loader: 'olsson', config_name: 'vggt', splat_implementation: 'gsplat' },
  models: [{ id: 'vggt', label: 'VGGT', capabilities: { iterative_splat: true, mvs: false, share_intrinsics: true } }],
  loaders: ['olsson'], loader_options: { olsson: [] }, graph_partitioners: [], global_descriptors: [],
  retrievers: [], correspondence_generators: [], verifiers: [], log_levels: ['INFO'], gaussian_splatting_models: ['base_gs'],
  splat_implementations: [{ id: 'none', label: 'No splats' }, { id: 'gsplat', label: 'Gaussian splats' }],
};
const sample = { id: 'door', label: 'Door', image_count: 12, description: 'Test scene',
  recommendations: { loader: 'olsson', config_name: 'vggt', max_resolution: 518 } };
const emptyCatalog = { base_dir: '/results', items: [], directory_modified_at: {} };
const defaults = {
  '/api/configuration': config,
  '/api/configuration/parameters': { model: { fields: [] }, gaussian: { fields: [] } },
  '/api/hardware': { devices: [{ id: 'cuda:0', kind: 'cuda', label: 'Test GPU', status: 'available', supports_gaussian_splatting: true }], summary: 'Test GPU' },
  '/api/setup': { status: 'ready', summary: 'Ready', counts: { ready: 0, warning: 0, error: 0, optional: 0 }, items: [] },
  '/api/samples': { items: [sample] }, '/api/jobs': { items: [] }, '/api/scenes': emptyCatalog,
  '/api/photos': { items: [] },
};
let server, browser, baseUrl;
before(async () => {
  server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname.startsWith('/api/')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(defaults[url.pathname] || {})); return;
    }
    const file = url.pathname === '/' ? path.join(root, 'templates/index.html') : path.join(root, url.pathname);
    if (!file.startsWith(root + path.sep)) { res.writeHead(404); res.end(); return; }
    try {
      const data = await fs.readFile(file);
      res.writeHead(200, { 'Content-Type': file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.png') ? 'image/png' : 'text/html' });
      res.end(data);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
    args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
});
after(async () => { await browser?.close(); await new Promise(resolve => server?.close(resolve)); });

async function pageFor(t) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, reducedMotion: 'reduce' });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.routeWebSocket('**/api/events/**', () => {});
  t.after(async () => { await page.close(); assert.deepEqual(errors, [], 'No uncaught browser errors'); });
  return page;
}
async function ready(page, suffix = '') {
  await page.goto(baseUrl + suffix);
  await page.waitForFunction(() => !!window.gtsfmViewer && !document.getElementById('studio-startup'));
}
async function eventually(predicate) {
  const deadline = Date.now() + 10000;
  while (!predicate()) { assert.ok(Date.now() < deadline, 'Condition timed out'); await delay(25); }
}
function item(rel_path, revision = '1') {
  return { kind: 'scene', rel_path, label: rel_path, point_count: 1, camera_count: 0,
    modified_at: 1, revision, points: '/data/points3D.txt', images: null, splats: null };
}

test('controls work during delayed and failed result discovery; retry recovers the catalog', async t => {
  const page = await pageFor(t);
  let request;
  await page.route('**/api/scenes', route => { request = route; });
  await ready(page);
  await eventually(() => !!request);
  await page.locator('#orthographicBtn').click();
  await page.waitForFunction(() => window.gtsfmViewer.getState().orthographic);
  assert.equal(await page.locator('#orthographicBtn').getAttribute('aria-pressed'), 'true');
  await request.fulfill({ status: 500, json: { error: 'Test error' } });
  await page.getByRole('tab', { name: 'Results', exact: true }).click();
  await page.getByText('Could not load results (500).', { exact: false }).waitFor();
  await page.locator('#orthographicBtn').click();
  await page.waitForFunction(() => !window.gtsfmViewer.getState().orthographic);
  request = null;
  await page.getByRole('button', { name: 'Refresh results', exact: true }).click();
  await eventually(() => !!request);
  await request.fulfill({ json: emptyCatalog });
  await page.getByText('No results found', { exact: true }).waitFor();
});

test('an explicit GPU choice survives recommendation and schema updates', async t => {
  const page = await pageFor(t); await ready(page);
  await page.getByRole('button', { name: 'GTSFM samples', exact: true }).click();
  await page.locator('label').filter({ hasText: /^Sample scene/ }).locator('select').selectOption('door');
  await page.getByRole('button', { name: 'Remote VM', exact: true }).click();
  const gpu = page.locator('label').filter({ hasText: /^Modal VM GPU/ }).locator('select');
  await gpu.selectOption('H100');
  await page.locator('label').filter({ hasText: /^Splat implementation/ }).locator('select').selectOption('none');
  const refreshed = page.waitForResponse('**/api/configuration');
  await page.evaluate(() => window.dispatchEvent(new Event('focus'))); await refreshed;
  await page.waitForFunction(() => !document.querySelector('.configuration-unavailable .spin'));
  assert.equal(await gpu.inputValue(), 'H100');
});

test('result refresh preserves root-first ordering and reloads changed point colors', async t => {
  const page = await pageFor(t);
  let revision = '1', pointRequests = [];
  await page.route('**/api/scenes', route => route.fulfill({ json: { ...emptyCatalog,
    items: [item('old/root'), item('new/child/reconstruction'), item('new/root', revision)],
    directory_modified_at: { old: 1, new: 3, 'new/child': 4 },
  } }));
  await page.route('**/data/points3D.txt?*', route => {
    pointRequests.push(route.request().url());
    return route.fulfill({ body: revision === '1' ? '1 0 0 0 255 0 0\n' : '1 10 20 30 0 0 255\n', contentType: 'text/plain' });
  });
  await ready(page); await page.getByRole('tab', { name: 'Results', exact: true }).click();
  assert.deepEqual(await page.locator('#sceneList .result-open').evaluateAll(nodes => nodes.map(n => n.title)), ['new/root', 'new/child/reconstruction', 'old/root']);
  const open = page.locator('.result-open[title="new/root"]'); await open.click();
  await page.waitForFunction(() => window.gtsfmViewer.getState().points === 1 && !window.gtsfmViewer.isBusy());
  revision = '2';
  const refreshed = page.waitForResponse('**/api/scenes');
  await page.getByRole('button', { name: 'Refresh results', exact: true }).click(); await refreshed;
  await page.getByRole('button', { name: 'Refresh results', exact: true }).waitFor();
  await open.click();
  await page.waitForFunction(() => !window.gtsfmViewer.isBusy() && window.gtsfmViewer.parsedPointsCache.has('/data/points3D.txt?v=2'));
  assert.equal(pointRequests.length, 2);
  const point = await page.evaluate(() => window.gtsfmViewer.parsedPointsCache.get('/data/points3D.txt?v=2')[0]);
  assert.deepEqual(point, { x: 10, y: 30, z: 20, r: 0, g: 0, b: 1 });
});

test('both job streams reconnect and selected-job updates resume', async t => {
  const page = await pageFor(t);
  const job = { id: 'job-1', name: 'Test run', status: 'running', spec: { splat_implementation: 'gsplat' }, log_tail: [], command: [] };
  const sockets = { list: [], detail: [] };
  await page.route('**/api/jobs', route => route.fulfill({ json: { items: [job] } }));
  await page.routeWebSocket('**/api/events/jobs', ws => { sockets.list.push(ws); ws.send(JSON.stringify({ items: [job] })); });
  await page.routeWebSocket('**/api/events/jobs/job-1', ws => { sockets.detail.push(ws); ws.send(JSON.stringify({ job, live: { stage: 'pipeline', message: 'Before reconnect' } })); });
  await ready(page); await eventually(() => sockets.list.length === 1 && sockets.detail.length === 1);
  await page.getByText('Before reconnect', { exact: true }).waitFor();
  await sockets.list[0].close({ code: 1011 }); await sockets.detail[0].close({ code: 1011 });
  await page.getByText('Connection interrupted. Reconnecting to live updates…', { exact: true }).waitFor();
  await eventually(() => sockets.list.length === 2 && sockets.detail.length === 2);
  sockets.detail[1].send(JSON.stringify({ job, live: { stage: 'pipeline', message: 'After reconnect' } }));
  await page.getByText('After reconnect', { exact: true }).waitFor();
  await page.waitForFunction(() => !document.querySelector('.connection-notice'));
});

test('completed runs refresh the result catalog without a page reload', async t => {
  const page = await pageFor(t); let socket, scans = 0;
  await page.route('**/api/scenes', route => { scans++; return route.fulfill({ json: emptyCatalog }); });
  await page.routeWebSocket('**/api/events/jobs', ws => { socket = ws; });
  await ready(page); await eventually(() => socket && scans === 1);
  socket.send(JSON.stringify({ items: [{ id: 'done', name: 'Completed', status: 'completed', spec: {}, command: [], log_tail: [] }] }));
  await eventually(() => scans === 2);
});

test('renderer disposal cancels downloads and releases timers, observers, and engines', async t => {
  const page = await pageFor(t); await ready(page);
  let request;
  await page.route('**/pending-points', route => { request = route; });
  await page.evaluate(() => { window.pendingLoad = window.gtsfmViewer.loadScene({ pointsUrl: '/pending-points' }); });
  await eventually(() => !!request);
  const result = await page.evaluate(async () => {
    const old = window.gtsfmViewer;
    let timerCleared = false, observerDisconnected = false, resizeRemoved = false;
    const clear = window.clearInterval, remove = window.removeEventListener.bind(window), disconnect = old.resizeObserver.disconnect.bind(old.resizeObserver);
    window.clearInterval = id => { if (id === old.photoTimer) timerCleared = true; clear(id); };
    window.removeEventListener = (type, listener, options) => { if (listener === old.onResize) resizeRemoved = true; remove(type, listener, options); };
    old.resizeObserver.disconnect = () => { observerDisconnected = true; disconnect(); };
    old.dispose(); old.dispose();
    const loaded = await window.pendingLoad;
    const afterDisposal = window.BABYLON.Engine.Instances.length;
    const next = window.createGtsfmViewer(document.getElementById('renderCanvas'));
    next.toggleOrthographicProjection();
    const remounted = next.getState().orthographic;
    next.dispose();
    return { loaded, afterDisposal, remounted, finalEngines: window.BABYLON.Engine.Instances.length, timerCleared, observerDisconnected, resizeRemoved };
  });
  assert.deepEqual(result, { loaded: false, afterDisposal: 0, remounted: true, finalEngines: 0, timerCleared: true, observerDisconnected: true, resizeRemoved: true });
});

test('React controls preserve projection, background, and plane preferences across reload', async t => {
  const page = await pageFor(t); await ready(page);
  await page.locator('#orthographicBtn').click();
  await page.locator('#backgroundSelect').selectOption('light-gray');
  await page.locator('#toggleGround').click();
  await page.locator('#groundY').fill('2.4');
  await page.locator('#toggleStats').click();
  assert.equal(await page.locator('#sceneStats').isVisible(), false);
  await ready(page);
  assert.equal(await page.locator('#orthographicBtn').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#backgroundSelect').inputValue(), 'light-gray');
  assert.equal(await page.locator('#toggleGround').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#groundY').inputValue(), '2.4');
  for (const [axis, expected] of Object.entries({ x: [1, 0, 0], '-x': [-1, 0, 0], y: [0, 0, 1], '-y': [0, 0, -1], z: [0, 1, 0], '-z': [0, -1, 0] })) {
    await page.locator(`[data-view-axis="${axis}"]`).click();
    const actual = await page.evaluate(() => {
      const camera = window.gtsfmViewer.camera; camera.getViewMatrix(true);
      return camera.position.subtract(camera.target).normalize().asArray();
    });
    actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-5));
  }
});

test('switching selected jobs closes the old stream and ignores its pending reconnect', async t => {
  const page = await pageFor(t);
  const first = { id: 'one', name: 'First job', status: 'running', spec: {}, command: [], log_tail: [] };
  const second = { ...first, id: 'two', name: 'Second job' };
  const sockets = [];
  await page.route('**/api/jobs', route => route.fulfill({ json: { items: [first, second] } }));
  await page.routeWebSocket('**/api/events/jobs/*', ws => { sockets.push(ws); });
  await ready(page); await eventually(() => sockets.length === 1);
  await sockets[0].close({ code: 1011 });
  await page.getByRole('tab', { name: /Activity/ }).click();
  await page.locator('.job-card').filter({ hasText: 'Second job' }).getByRole('button', { name: 'View progress', exact: true }).click();
  await eventually(() => sockets.length === 2);
  assert.ok(sockets[1].url().endsWith('/two'));
  // Advance beyond the old stream's retry deadline; it must stay closed.
  await delay(1200);
  assert.equal(sockets.length, 2);
});
