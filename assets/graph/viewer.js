  const THEMES = __THEMES__;
  const L = __LABELS__;
  const TYPE_NAMES = __TYPE_NAMES__;
  let isDark = __START_DARK__;
  const EDGE_TOTAL = __EDGE_TOTAL__;   // real relations; the slider keeps the N most important ones
  let tidyPositions = [];

  document.body.insertAdjacentHTML('beforeend', `
    <div class="zoom-controls">
      <button class="zoom-btn" id="zoom-in" title="${L.zoom_in}" aria-label="${L.zoom_in}">
        <svg viewBox="0 0 16 16"><path d="M8 2.5v11M2.5 8h11"/></svg></button>
      <button class="zoom-btn" id="zoom-out" title="${L.zoom_out}" aria-label="${L.zoom_out}">
        <svg viewBox="0 0 16 16"><path d="M2.5 8h11"/></svg></button>
      <button class="zoom-btn" id="zoom-fit" title="${L.fit}" aria-label="${L.fit}">
        <svg viewBox="0 0 16 16"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg></button>
    </div>
    <div class="action-buttons">
      <button id="theme-btn" class="float-btn"></button>
      <button id="tidy-btn" class="float-btn"></button>
      <label class="importance" id="importance" hidden>
        <span>${L.importance}</span>
        <input type="range" id="importance-range" min="1" max="${EDGE_TOTAL}" value="${EDGE_TOTAL}">
        <span class="count" id="importance-count"></span>
      </label>
    </div>
    <div id="side-panel">
      <div class="panel-header"><h2 id="panel-title"></h2><button class="close-btn" id="close-btn">&times;</button></div>
      <div id="panel-badge"></div>
      <div class="panel-section"><h3>${L.overview}</h3><div class="panel-box" id="panel-overview"></div></div>
      <div class="panel-section"><h3>${L.connected}</h3><ul class="connection-list" id="panel-connections"></ul></div>
    </div>`);

  function applyTheme() {
    const theme = isDark ? THEMES.dark : THEMES.light;
    document.body.classList.toggle('dark-mode', isDark);
    document.body.style.backgroundColor = theme.bg;
    const container = document.getElementById('mynetwork');
    if (container) container.style.backgroundColor = theme.bg;

    nodes.update(nodes.get().map(node => {
      const c = theme.nodes[node.entity_type] || theme.nodes['Entity'];
      return { id: node.id,
               color: { background: c.background, border: c.border,
                        highlight: { background: c.background, border: theme.highlight } },
               font: { color: theme.text } };
    }));
    edges.update(edges.get().map(edge => ({
      id: edge.id,
      color: { color: edge.dashes === true ? theme.edge_virtual : theme.edge, highlight: theme.edge_highlight },
      font: { color: theme.edge_text, strokeColor: theme.bg }
    })));
    document.getElementById('theme-btn').textContent = isDark ? L.light_mode : L.dark_mode;
  }

  document.getElementById('theme-btn').addEventListener('click', () => { isDark = !isDark; applyTheme(); });
  function zoomBy(factor) {   // zoom around the current view centre
    const scale = Math.min(4, Math.max(0.08, network.getScale() * factor));
    network.moveTo({ scale: scale, animation: { duration: 220, easingFunction: 'easeInOutQuad' } });
  }
  document.getElementById('zoom-in').addEventListener('click', () => zoomBy(1.3));
  document.getElementById('zoom-out').addEventListener('click', () => zoomBy(1 / 1.3));
  // Fit the visible nodes into the canvas, leaving room for the button bar at the bottom
  function fitView(duration) {
    const ids = nodes.get().filter(n => !n.hidden).map(n => n.id);
    if (!ids.length) return;
    const pos = network.getPositions(ids);
    const xs = ids.map(id => pos[id].x), ys = ids.map(id => pos[id].y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const margin = 70, padTop = 24, padBottom = 96, padSide = 24;   // margin: node radius + label, in graph units
    const box = document.getElementById('mynetwork');
    const scale = Math.min(1.4, (box.clientWidth - 2 * padSide) / Math.max(x1 - x0 + 2 * margin, 1),
                                 (box.clientHeight - padTop - padBottom) / Math.max(y1 - y0 + 2 * margin, 1));
    network.moveTo({ position: { x: (x0 + x1) / 2, y: (y0 + y1) / 2 + (padBottom - padTop) / 2 / scale }, scale: scale,
                     animation: duration ? { duration: duration, easingFunction: 'easeInOutQuad' } : false });
  }
  document.getElementById('zoom-fit').addEventListener('click', () => fitView(400));

  document.getElementById('tidy-btn').textContent = L.tidy;
  document.getElementById('tidy-btn').addEventListener('click', () => {   // put every node back where the layout placed it
    tidyPositions.forEach(p => network.moveNode(p.id, p.x, p.y));
    fitView(600);
  });

  // Show only the `limit` most important relations; nodes with no visible relation disappear,
  // the hub stays, and implicit connectors only show while their cluster is visible. Positions never change.
  function applyImportance(limit) {
    const keep = new Set(nodes.get().filter(n => n.entity_type === 'Topic').map(n => n.id));
    edges.get().forEach(e => { if (!e.kg_virtual && e.kg_rank <= limit) { keep.add(e.from); keep.add(e.to); } });
    nodes.update(nodes.get().map(n => ({ id: n.id, hidden: !keep.has(n.id) })));
    edges.update(edges.get().map(e => ({
      id: e.id, hidden: e.kg_virtual ? !(keep.has(e.from) && keep.has(e.to)) : e.kg_rank > limit })));
    document.getElementById('importance-count').textContent =
      L.showing.replace('{shown}', Math.min(limit, EDGE_TOTAL)).replace('{total}', EDGE_TOTAL);
    const selected = network.getSelectedNodes();
    if (selected.length && !keep.has(selected[0])) closePanel();
  }
  const importanceBox = document.getElementById('importance');
  const importanceRange = document.getElementById('importance-range');
  importanceBox.hidden = EDGE_TOTAL < 4;   // nothing worth filtering in a tiny graph
  importanceRange.addEventListener('input', () => applyImportance(+importanceRange.value));
  importanceRange.addEventListener('change', () => fitView(500));   // on release: zoom to what is left

  function closePanel() {
    document.getElementById('side-panel').classList.remove('open');
    network.unselectAll();
  }
  document.getElementById('close-btn').addEventListener('click', closePanel);

  function focusAndOpenPanel(nodeId) {
    const node = nodes.get(nodeId);
    if (!node) return;
    network.focus(nodeId, { scale: 1.2, animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
    network.setSelection({ nodes: [nodeId] });

    document.getElementById('panel-title').textContent = String(nodeId);   // the id is the unwrapped name
    const type = node.entity_type || 'Entity';
    const theme = isDark ? THEMES.dark : THEMES.light;
    const swatch = (theme.nodes[type] || theme.nodes['Entity']).background;
    const badge = document.getElementById('panel-badge');
    badge.innerHTML = '<span class="info-tag"><i style="background:' + swatch + '"></i></span>';
    badge.firstChild.appendChild(document.createTextNode(TYPE_NAMES[type] || type));
    document.getElementById('panel-overview').innerHTML = node.panel_html || L.no_context;

    const list = document.getElementById('panel-connections');
    list.innerHTML = '';
    const connected = network.getConnectedNodes(nodeId).filter(id => !(nodes.get(id) || {}).hidden);
    if (connected.length === 0) {
      const li = document.createElement('li');
      li.style.cssText = 'list-style:none;color:var(--text-muted);font-size:13.5px';
      li.textContent = L.no_connections;
      list.appendChild(li);
    }
    connected.forEach(id => {
      const other = nodes.get(id);
      const li = document.createElement('li');
      li.className = 'connection-item';
      li.textContent = String(id);   // textContent: article text is never parsed as HTML
      li.addEventListener('click', () => focusAndOpenPanel(id));
      list.appendChild(li);
    });
    document.getElementById('side-panel').classList.add('open');
  }

  setTimeout(function () {
    tidyPositions = nodes.get().map(n => ({ id: n.id, x: n.x, y: n.y }));
    fitView(0);
    applyImportance(EDGE_TOTAL);
    applyTheme();
    network.on('click', params => params.nodes.length > 0 ? focusAndOpenPanel(params.nodes[0]) : closePanel());
  }, 700);
