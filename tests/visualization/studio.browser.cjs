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
  '/api/execution-configurations': { items: [] },
  '/api/modal/account': {status:'disconnected', profile:'', workspace:'', profiles:['native-profile'], login_url:'', code:'', error:''},
};
let server, browser, baseUrl;
before(async () => {
  server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname.startsWith('/api/')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      const experiment = url.pathname.startsWith('/api/experiments/') ? {job:{id:url.pathname.split('/').pop(),name:'Fixture'},provenance:{settings:{},source:null,dataset:null},metrics:{},results:[],preferred_result:null} : null;
      res.end(JSON.stringify(defaults[url.pathname] || experiment || {})); return;
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
async function step(page, label) {
  if (label === 'Compute') {const disclosure = page.locator('.saved-execution'); if (!await disclosure.getAttribute('open')) await disclosure.locator(':scope > summary').click();}
}
async function openResults(page) {
  await page.getByRole('tab', {name:'Experiments', exact:true}).click();
  await page.getByRole('button', {name:'Open existing reconstruction', exact:true}).click();
}
async function chooseCapture(page) {
  await step(page, 'Images');
  await page.getByRole('button', { name: 'Sample datasets', exact: true }).click();
  await page.getByLabel('Sample scene', { exact: true }).selectOption('door');
}
function item(rel_path, revision = '1') {
  return { kind: 'scene', rel_path, label: rel_path, point_count: 1, camera_count: 0,
    modified_at: 1, revision, points: '/data/points3D.txt', images: null, splats: null };
}

test('controls work during delayed and failed result discovery; retry recovers the catalog', async t => {
  const page = await pageFor(t);
  let request;
  await page.route('**/api/scenes', route => { request = route; });
  await ready(page, '?view=results');
  await eventually(() => !!request);
  await page.locator('#viewControls > summary').click();
  await page.locator('#orthographicBtn').click();
  await page.waitForFunction(() => window.gtsfmViewer.getState().orthographic);
  assert.equal(await page.locator('#orthographicBtn').getAttribute('aria-pressed'), 'true');
  await request.fulfill({ status: 500, json: { error: 'Test error' } });
  await openResults(page);
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
  await page.getByRole('button', { name: 'Sample datasets', exact: true }).click();
  await page.locator('label').filter({ hasText: /^Sample scene/ }).locator('select').selectOption('door');
  await step(page, 'Compute');
  await page.getByRole('button', { name: /Modal cloud/ }).click();
  const gpu = page.locator('label').filter({ hasText: /^GPU/ }).locator('select');
  await gpu.selectOption('H100');
  await step(page, 'Pipeline');
  await page.locator('label').filter({ hasText: /^Splat implementation/ }).locator('select').selectOption('none');
  const refreshed = page.waitForResponse('**/api/configuration');
  await page.evaluate(() => window.dispatchEvent(new Event('focus'))); await refreshed;
  await page.waitForFunction(() => !document.querySelector('.configuration-unavailable .spin'));
  await step(page, 'Compute');
  assert.equal(await gpu.inputValue(), 'H100');
});

test('invalid saved-configuration responses show a load error without crashing the form', async t => {
  const page = await pageFor(t);
  let response;
  await page.route('**/api/execution-configurations', route => route.fulfill({ json: response }));
  for (const invalid of [{}, { items: null }, { items: [null] }]) {
    response = invalid;
    await ready(page);
    await step(page, 'Compute');
    await page.getByText('Unable to load saved execution configurations:', { exact: false }).waitFor();
    assert.equal(await page.getByLabel('Saved execution configuration', { exact: true }).isEnabled(), true);
    await step(page, 'Compute');
  await page.getByRole('button', { name: /Modal cloud/ }).click();
    await page.getByRole('heading', {name: 'Connect your Modal account'}).waitFor();
  }
});

async function fixtureModalOnboarding(page) {
  await page.route('**/api/modal/connect', route => route.fulfill({json:{status:'connected',profile:'native-profile',workspace:'Research',profiles:['native-profile'],login_url:'',code:'',error:''}}));
  await page.route('**/api/modal/deploy', route => {
    const requested = route.request().postDataJSON();
    assert.equal(requested.profile, 'native-profile');
    assert.equal('token_secret' in requested, false);
    return route.fulfill({ json: { id: 'fixture-deployment', status: 'completed', phase: 'ready', stage: 'Ready',
      gpu: requested.gpu, cpu: requested.cpu, memory_mb: requested.memory_mb,
      endpoint: 'https://fixture.modal.test', api_key: 'fixture-workspace-key', log_tail: [] } });
  });
  await page.route('**/api/remote/inspect', route => route.fulfill({json:{hardware:defaults['/api/hardware'],configuration:config}}));
  await page.getByRole('button', {name:'Connect saved Modal account'}).click();
  await page.getByRole('heading', {name:'Connected to Research'}).waitFor();
  assert.equal(await page.getByLabel('Token secret', {exact:true}).count(),0);
}

test('a fresh Modal user deploys with the GPU device displayed and submitted consistently', async t => {
  const page = await pageFor(t);
  await ready(page);
  await page.getByRole('button', { name: 'Sample datasets', exact: true }).click();
  await page.getByLabel('Sample scene', { exact: true }).selectOption('door');
  await step(page, 'Pipeline');
  await page.getByLabel('Splat implementation', { exact: true }).selectOption('none');
  await step(page, 'Compute');
  await page.getByRole('button', { name: /Modal cloud/ }).click();
  await fixtureModalOnboarding(page);
  assert.equal(await page.getByLabel('Remote hardware', { exact: true }).count(), 0);
  let submitted;
  await page.route('**/api/jobs', route => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { items: [] } });
    submitted = route.request().postDataJSON();
    return route.fulfill({ status: 202, json: { id: 'fixture-job', name: 'Fixture', status: 'queued', spec: submitted,
      command: [], log_tail: [] } });
  });
  await page.getByRole('button', { name: 'Run experiment', exact: true }).click();
  await eventually(() => !!submitted);
  assert.equal(submitted.hardware, 'cuda:0');
});

test('a saved Modal configuration normalizes its legacy CPU choice to the allocated GPU', async t => {
  const page = await pageFor(t);
  await page.route('**/api/execution-configurations', route => route.fulfill({json:{items:[{name:'Saved CPU',settings:{execution_target:'remote',remote_connection:'api',remote_provider:'modal',remote_hardware:'cpu'}}]}}));
  await ready(page); await chooseCapture(page); await step(page,'Compute');
  await page.getByLabel('Saved execution configuration', {exact:true}).selectOption('Saved CPU');
  await fixtureModalOnboarding(page);
  assert.equal(await page.getByText('This Modal workspace runs on cuda:0.', {exact:false}).count(),0);
  assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isEnabled(),true);
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
  await ready(page); await openResults(page);
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
  await page.getByRole('tab', { name: 'Experiments', exact: false }).click();
  await page.getByRole('button', { name: 'View progress', exact: true }).click();
  await page.locator('.experiment-progress').getByText('Before reconnect', { exact: true }).waitFor();
  await sockets.list[0].close({ code: 1011 }); await sockets.detail[0].close({ code: 1011 });
  await page.getByText('Connection interrupted. Reconnecting to live updates…', { exact: true }).waitFor();
  await eventually(() => sockets.list.length === 2 && sockets.detail.length === 2);
  sockets.detail[1].send(JSON.stringify({ job, live: { stage: 'pipeline', message: 'After reconnect' } }));
  await page.locator('.experiment-progress').getByText('After reconnect', { exact: true }).waitFor();
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
  const page = await pageFor(t); await ready(page, '?view=results');
  await page.locator('#viewControls > summary').click();
  await page.locator('#hud .display-settings > summary').click();
  await page.locator('#orthographicBtn').click();
  await page.locator('#backgroundSelect').selectOption('light-gray');
  await page.locator('#toggleGround').click();
  await page.locator('#groundY').fill('2.4');
  await page.locator('#toggleStats').click();
  assert.equal(await page.locator('#sceneStats').isVisible(), false);
  await ready(page, '?view=results');
  await page.locator('#viewControls > summary').click();
  await page.locator('#hud .display-settings > summary').click();
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
  await page.getByRole('tab', { name: /Experiments/ }).click();
  await page.locator('.job-card').filter({ hasText: 'Second job' }).getByRole('button', { name: 'View progress', exact: true }).click();
  await eventually(() => sockets.length === 2);
  assert.ok(sockets[1].url().endsWith('/two'));
  // Advance beyond the old stream's retry deadline; it must stay closed.
  await delay(1200);
  assert.equal(sockets.length, 2);
});


test('one-page setup preserves inputs across navigation and submits only through Run', async t => {
  const page=await pageFor(t); let posts=0;
  await page.route('**/api/jobs',route=>{if(route.request().method()==='POST') posts++; return route.fulfill({json:{items:[]}});});
  await ready(page); assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isDisabled(),true);
  await chooseCapture(page); await page.getByLabel('Experiment name',{exact:true}).fill('my-capture');
  await page.getByRole('button',{name:/Modal cloud/}).click();
  await page.locator('#computeTarget').getByRole('button',{name:/This computer/}).click();
  assert.equal(posts,0);
  await page.getByRole('tab',{name:'Experiments',exact:true}).click();
  await page.getByRole('tab',{name:'New experiment',exact:true}).click();
  assert.equal(await page.getByLabel('Experiment name',{exact:true}).inputValue(),'my-capture');
  assert.equal(await page.getByLabel('Sample scene',{exact:true}).inputValue(),'door');
});

test('mobile setup keeps a readable Run action inside the viewport without overflow', async t=>{
  const page=await pageFor(t); await page.setViewportSize({width:390,height:844}); await ready(page); await chooseCapture(page);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
  const button=page.getByRole('button',{name:'Run experiment',exact:true}); const rect=await button.boundingBox();
  assert.ok(rect.y>=126 && rect.y+rect.height<=844);
  for(const selector of ['.folder-label','.target-choice','.primary-action','.section-heading small']) {
    const sizes=await page.locator(selector).evaluateAll(nodes=>nodes.filter(n=>n.getClientRects().length).map(n=>parseFloat(getComputedStyle(n).fontSize)));
    assert.ok(sizes.every(size=>size>=16),`${selector} must remain readable`);
  }
});

test('environment checks open on demand and Escape returns focus to the trigger', async t => {
  const page = await pageFor(t); await ready(page);
  assert.equal(await page.getByRole('region', { name: 'Setup checks', exact: true }).count(), 0);
  const trigger = page.getByRole('button', { name: 'This computer ready', exact: true });
  await trigger.click();
  await page.getByRole('region', { name: 'Setup checks', exact: true }).waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await trigger.getAttribute('aria-expanded'), 'false');
  assert.equal(await trigger.evaluate(node => document.activeElement === node), true);
});


test('workspace navigation supports horizontal keyboard movement at all sizes', async t=>{
 const page=await pageFor(t); await ready(page);
 for(const width of [1600,390]) {await page.setViewportSize({width,height:844});
 await page.getByRole('tab',{name:'New experiment',exact:true}).focus(); await page.keyboard.press('ArrowRight');
 await page.waitForFunction(()=>document.querySelector('.studio-tab[value=activity]')?.getAttribute('aria-selected')==='true' || [...document.querySelectorAll('[role=tab]')].some(n=>n.textContent==='Experiments' && n.getAttribute('aria-selected')==='true'));
 await page.keyboard.press('ArrowLeft'); await page.getByRole('heading',{name:'New experiment',exact:true}).waitFor();}
});

function experimentFixture(id = 'experiment-one', datasetHash = 'same-input') {
  const result = item(`${id}/merged_retriangulated`);
  return {job:{id,name:id,status:'completed'},provenance:{settings:{name:id,sample_id:'door',loader:'olsson',config_name:'vggt',splat_implementation:'none',hydra_overrides:{}},source:{commit:'1234567890',dirty:true,fingerprint:'code'},dataset:{manifest_sha256:datasetHash,files:[{path:'one.jpg'}]}},configuration:{model:{threshold:0.5}},metrics:{optimization:{error:0,cameras:4}},results:[result],preferred_result:result};
}
async function completedExperiment(page, {photos=false,splats=false}={}) {
  const details=experimentFixture(); const job={...details.job,spec:details.provenance.settings,command:[],log_tail:[]};
  if (photos) {
    details.preferred_result.images='/data/images.txt';
    details.preferred_result.camera_count=1;
    await page.route('**/data/images.txt*',route=>route.fulfill({body:'1 1 0 0 0 0 0 -3 1 photo.jpg\n\n',contentType:'text/plain'}));
    await page.route('**/data/cameras.txt*',route=>route.fulfill({body:'1 PINHOLE 800 600 400 400 400 300\n',contentType:'text/plain'}));
    await page.route('**/api/photos?*',route=>route.fulfill({json:{items:[{name:'photo.jpg',thumbnail:'/data/photo.png',preview:'/data/photo.png'}]}}));
    const photo=await fs.readFile(path.join(root,'static/brand/bee-favicon.png'));
    await page.route('**/data/photo.png',route=>route.fulfill({body:photo,contentType:'image/png'}));
  }
  if (splats) {
    details.preferred_result.splats='/data/splats.ply';
    const properties=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
    const header=['ply','format binary_little_endian 1.0','element vertex 1',...properties.map(p=>'property float '+p),'end_header',''].join('\n');
    const vertex=Buffer.alloc(properties.length*4);
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0].forEach((value,index)=>vertex.writeFloatLE(value,index*4));
    const body=Buffer.concat([Buffer.from(header),vertex]);
    await page.route('**/data/splats.ply*',route=>route.fulfill({body,contentType:'application/octet-stream'}));
  }
  await page.route('**/api/jobs',route=>route.fulfill({json:{items:[job]}}));
  await page.route('**/api/experiments/experiment-one',route=>route.fulfill({json:details}));
  await page.route('**/data/points3D.txt*',route=>route.fulfill({body:'1 0 0 0 255 0 0\n',contentType:'text/plain'}));
  await ready(page); await page.getByRole('tab',{name:'Experiments',exact:true}).click();
  await page.locator('.job-card').getByRole('button',{name:'View details',exact:true}).click();
  return details;
}

test('source photos remain clickable beside viewer controls at narrow and wide sizes',async t=>{
  const page=await pageFor(t);
  await page.setViewportSize({width:747,height:972});
  await completedExperiment(page,{photos:true});
  for (const width of [747,390,914,1440]) {
    await page.setViewportSize({width,height:972});
    await page.getByRole('button',{name:'Expand photos',exact:true}).click();
    await page.getByRole('button',{name:'View camera 1: photo.jpg',exact:true}).click();
    await page.getByRole('button',{name:'Enlarge photo.jpg',exact:true}).click();
    const dialog=page.getByRole('dialog',{name:'photo.jpg',exact:true});
    await dialog.waitFor();
    await page.waitForFunction(()=>document.querySelector('.photo-lightbox img')?.naturalWidth>0);
    await page.getByRole('button',{name:'Close photo',exact:true}).click();
    await page.getByRole('button',{name:'Collapse photos',exact:true}).click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  }
});

test('a completed experiment switches between splats and its cameras, points and photos',async t=>{
  const page=await pageFor(t);
  await completedExperiment(page,{photos:true,splats:true});
  await page.waitForFunction(()=>window.gtsfmViewer.getState().cameras===1 && window.gtsfmViewer.getState().points===1);
  const choice=page.getByLabel('Reconstruction view',{exact:true});
  assert.equal(await choice.inputValue(),'scene');
  await choice.selectOption('splat');
  await page.waitForFunction(()=>window.gtsfmViewer.getState().mode==='splat' && window.gtsfmViewer.getState().splats===1 && !window.gtsfmViewer.isBusy());
  await choice.selectOption('scene');
  await page.waitForFunction(()=>window.gtsfmViewer.getState().mode==='scene' && window.gtsfmViewer.getState().cameras===1 && window.gtsfmViewer.getState().points===1 && !window.gtsfmViewer.isBusy());
  await page.getByRole('button',{name:'View camera 1: photo.jpg',exact:true}).waitFor();
});

test('an experiment opens its own geometry and reuses settings without submitting another run',async t=>{
  const page=await pageFor(t); let posts=0;
  const details=await completedExperiment(page);
  await page.waitForFunction(()=>window.gtsfmViewer.getState().points===1 && !window.gtsfmViewer.isBusy());
  await page.getByText(/12345678.*local changes/).waitFor();
  await page.route('**/api/jobs',route=>{if(route.request().method()==='POST')posts++;return route.fulfill({json:{items:[]}});});
  await page.getByRole('button',{name:'Use these settings',exact:true}).click();
  await page.getByText('Settings loaded. Review the images and compute, then run when ready.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Experiment name',{exact:true}).inputValue(),details.job.name+'-copy');
  assert.equal(await page.getByLabel('Sample scene',{exact:true}).inputValue(),'door');
  assert.equal(posts,0);
});

test('refreshing an experiment loads the new revision of its geometry', async t => {
  const page = await pageFor(t);
  const details = await completedExperiment(page);
  await page.waitForFunction(() => window.gtsfmViewer.getState().points === 1);
  details.preferred_result.revision = '2';

  await page.route('**/data/points3D.txt?v=2', route => route.fulfill({
    body: '1 0 0 0 255 0 0\n2 1 1 1 0 255 0\n', contentType: 'text/plain',
  }));
  await page.getByRole('button', { name: 'Refresh experiment', exact: true }).click();
  await page.waitForFunction(() => window.gtsfmViewer.getState().points === 2);
});

test('comparison preserves measured zero and warns when datasets are not matched',async t=>{
  const page=await pageFor(t); const first=experimentFixture(),second=experimentFixture('experiment-two','different');second.metrics.optimization.error=0.25;
  await page.route('**/api/jobs',route=>route.fulfill({json:{items:[first,second].map(d=>({...d.job,spec:d.provenance.settings,log_tail:[],command:[]}))}}));
  await page.route('**/api/experiments/*',route=>route.fulfill({json:route.request().url().endsWith('experiment-two')?second:first}));
  await page.route('**/data/points3D.txt*',route=>route.fulfill({body:'1 0 0 0 255 0 0\n',contentType:'text/plain'}));
  await ready(page);await page.getByRole('tab',{name:'Experiments',exact:true}).click();
  await page.getByText('Metrics, configuration & comparison',{exact:true}).click();
  await page.getByLabel('Compare with',{exact:true}).selectOption('experiment-two');
  await page.getByText('Dataset identity differs or was not recorded. These runs are not a matched benchmark.',{exact:true}).waitFor();
  const row=page.getByRole('row').filter({hasText:'optimization / error'});
  assert.deepEqual(await row.locator('td').allTextContents(),['optimization / error','0','0.25']);
});

test('importing an experiment applies its pipeline but requires deliberate dataset selection and Run',async t=>{
  const page=await pageFor(t);let posts=0;
  await page.route('**/api/experiments/import',route=>route.fulfill({json:{settings:{name:'shared',loader:'olsson',config_name:'vggt',splat_implementation:'none'}}}));
  await page.route('**/api/jobs',route=>{if(route.request().method()==='POST')posts++;return route.fulfill({json:{items:[]}});});
  await ready(page);
  await page.getByLabel('Import experiment record',{exact:true}).setInputFiles({name:'experiment.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(experimentFixture().provenance))});
  await page.getByText('Settings loaded. Review the images and compute, then run when ready.',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Experiment name',{exact:true}).inputValue(),'shared-copy');
  assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isDisabled(),true);
  assert.equal(posts,0);
});

test('failed automatic Modal preparation never submits a reconstruction',async t=>{
  const page=await pageFor(t);let posts=0;
  await ready(page);await chooseCapture(page);await page.getByRole('button',{name:/Modal cloud/}).click();await fixtureModalOnboarding(page);
  await page.route('**/api/modal/deploy',route=>route.fulfill({json:{id:'failed',status:'failed',error:'Build could not finish',log_tail:[]}}));
  await page.route('**/api/jobs',route=>{if(route.request().method()==='POST')posts++;return route.fulfill({json:{items:[]}});});
  await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByText('Build could not finish',{exact:true}).first().waitFor();
  await page.getByRole('alert').filter({hasText:'Modal preparation did not finish'}).waitFor();
  assert.equal(posts,0);
  assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isEnabled(),true);
});

test('first-time Modal connection exposes a native browser link without deployment or credentials',async t=>{
  const page=await pageFor(t);let deployments=0;
  const pending={status:'signing_in',profile:'',workspace:'',profiles:[],login_url:'https://modal.com/token-flow/fixture',code:'ABCD',error:''};
  await page.route('**/api/modal/account',route=>route.fulfill({json:{...pending,status:'disconnected',login_url:'',code:''}}));
  await page.route('**/api/modal/connect',route=>route.fulfill({json:pending}));
  await page.route('**/api/modal/deploy',route=>{deployments++;return route.fulfill({json:{}});});
  await ready(page);await page.getByRole('button',{name:/Modal cloud/}).click();await page.getByRole('button',{name:'Sign in to Modal',exact:true}).click();
  assert.equal(await page.getByRole('link',{name:'Continue to Modal ↗',exact:true}).getAttribute('href'),pending.login_url);
  await page.getByText('ABCD',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Token secret',{exact:true}).count(),0);
  assert.equal(deployments,0);
});

test('mobile experiment inspection returns to the experiment list without losing its result',async t=>{
  const page=await pageFor(t);await page.setViewportSize({width:390,height:844});await completedExperiment(page);
  await page.getByRole('heading',{name:'experiment-one',exact:true}).waitFor();
  await page.getByRole('button',{name:'← All experiments',exact:true}).click();
  await page.locator('.job-card').getByRole('button',{name:'View details',exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
});

test('an experiment URL restores the selected result after reload',async t=>{
 const page=await pageFor(t);const details=experimentFixture();
 await page.route('**/api/jobs',route=>route.fulfill({json:{items:[{...details.job,spec:details.provenance.settings,log_tail:[],command:[]}]}}));
 await page.route('**/api/experiments/experiment-one',route=>route.fulfill({json:details}));
 await page.route('**/data/points3D.txt*',route=>route.fulfill({body:'1 0 0 0 255 0 0\n',contentType:'text/plain'}));
 await ready(page,'?experiment=experiment-one');
 await page.getByRole('heading',{name:'experiment-one',exact:true}).waitFor();
 assert.equal(await page.getByRole('tab',{name:'Experiments',exact:true}).getAttribute('aria-selected'),'true');
 await ready(page,'?experiment=experiment-one');await page.waitForFunction(()=>window.gtsfmViewer.getState().points===1);
});

test('import clears a previously selected local dataset instead of silently reusing it',async t=>{
 const page=await pageFor(t);const os=require('node:os');const folder=await fs.mkdtemp(path.join(os.tmpdir(),'gtsfm-import-input-'));await fs.writeFile(path.join(folder,'one.jpg'),Buffer.from([255,216,255,217]));t.after(()=>fs.rm(folder,{recursive:true,force:true}));
 const uploaded={id:'fixture-upload',name:'capture',path:'/managed/capture',file_count:1,bytes:4,analysis:{image_count:1}};
 await page.route('**/api/uploads',route=>route.fulfill({json:route.request().method()==='POST'?uploaded:{items:[uploaded]}}));
 await page.route('**/api/experiments/import',route=>route.fulfill({json:{settings:{name:'shared',config_name:'vggt',loader:'olsson',splat_implementation:'none'}}}));
 await ready(page);await page.locator('.folder-input').first().setInputFiles(folder);await page.getByRole('button',{name:/capture 1 file/}).waitFor();
 assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isEnabled(),true);
 await page.getByLabel('Import experiment record',{exact:true}).setInputFiles({name:'record.json',mimeType:'application/json',buffer:Buffer.from('{}')});
 await page.getByText('Settings loaded. Review the images and compute, then run when ready.',{exact:true}).waitFor();
 assert.equal(await page.getByRole('button',{name:'Run experiment',exact:true}).isDisabled(),true);
 await page.getByRole('button',{name:'Drop a folder here or click to choose one',exact:true}).waitFor();
});

test('importing after a sample does not substitute that sample for newly uploaded images',async t=>{
 const page=await pageFor(t);const os=require('node:os');const folder=await fs.mkdtemp(path.join(os.tmpdir(),'gtsfm-import-sample-'));await fs.writeFile(path.join(folder,'one.jpg'),Buffer.from([255,216,255,217]));t.after(()=>fs.rm(folder,{recursive:true,force:true}));
 const uploaded={id:'new-capture',name:'capture',path:'/managed/new-capture',file_count:1,bytes:4,analysis:{image_count:1}};
 await page.route('**/api/uploads',route=>route.fulfill({json:route.request().method()==='POST'?uploaded:{items:[uploaded]}}));
 await page.route('**/api/experiments/import',route=>route.fulfill({json:{settings:{name:'shared',sample_id:'unavailable',config_name:'vggt',loader:'olsson',splat_implementation:'none'}}}));
 let submitted;await page.route('**/api/jobs',route=>{if(route.request().method()!=='POST')return route.fulfill({json:{items:[]}});submitted=route.request().postDataJSON();return route.fulfill({json:{id:'new-job',name:'new',status:'queued',spec:submitted}})});
 await ready(page);await chooseCapture(page);await page.getByLabel('Import experiment record',{exact:true}).setInputFiles({name:'record.json',mimeType:'application/json',buffer:Buffer.from('{}')});
 await page.getByText('Settings loaded. Review the images and compute, then run when ready.',{exact:true}).waitFor();
 await page.locator('.folder-input').first().setInputFiles(folder);await page.getByRole('button',{name:/capture 1 file/}).waitFor();
 await page.getByRole('button',{name:'Run experiment',exact:true}).click();await eventually(()=>!!submitted);
 assert.equal(submitted.sample_id,'');assert.equal(submitted.dataset_dir,'/managed/new-capture');
});

test('switching experiments during a geometry download does not leave a false load error',async t=>{
 const page=await pageFor(t);const first=experimentFixture('first'),second=experimentFixture('second');
 first.results[0].points='/slow-first-points';second.results[0].points='/fast-second-points';let pending;
 await page.route('**/api/jobs',route=>route.fulfill({json:{items:[first,second].map(d=>({...d.job,spec:d.provenance.settings,log_tail:[],command:[]}))}}));
 await page.route('**/api/experiments/*',route=>route.fulfill({json:route.request().url().endsWith('second')?second:first}));
 await page.route('**/slow-first-points*',route=>{pending=route});await page.route('**/fast-second-points*',route=>route.fulfill({body:'1 0 0 0 255 0 0\n',contentType:'text/plain'}));
 await ready(page);await page.getByRole('tab',{name:'Experiments',exact:true}).click();await eventually(()=>!!pending);
 await page.locator('.job-card').filter({hasText:'second'}).getByRole('button',{name:'View details',exact:true}).click();
 await page.waitForFunction(()=>window.gtsfmViewer.getState().points===1&&!window.gtsfmViewer.isBusy());
 assert.equal(await page.locator('.experiment-detail .job-error').count(),0);
});
