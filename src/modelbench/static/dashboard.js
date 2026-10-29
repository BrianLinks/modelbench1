/*
 * ModelBench dashboard script.
 *
 * Small, dependency-free enhancements for pages the server already renders in full, so every page
 * works with this file blocked. It makes no network requests and never builds HTML from text: it
 * only reads attributes the server wrote and sets textContent, classes and hidden. The logic is in
 * plain functions at the top so it can be tested without a browser (tests/js/).
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else {
    root.ModelBenchUI = api;
  }
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', api.init);
    } else {
      api.init();
    }
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  var RUN_PATH = /^\/run\/[A-Za-z0-9._-]{1,64}$/;
  var HINT = 'Move over the chart for the value under the pointer.';

  // ---- logic without a DOM -------------------------------------------------------------------

  /** `{"accuracy": 0.9, ...}` from the text of a metrics cell: `accuracy=0.9000 f1=0.8000`. */
  function parseMetrics(text) {
    var out = {};
    String(text).split(/\s+/).forEach(function (part) {
      var at = part.lastIndexOf('=');
      if (at <= 0) return;
      var raw = part.slice(at + 1);
      if (NUMBER.test(raw)) out[part.slice(0, at)] = parseFloat(raw);
    });
    return out;
  }

  function isMissing(value) {
    return value === null || value === undefined || value === '' ||
      (typeof value === 'number' && isNaN(value));
  }

  /**
   * The order of `values` as an array of indices. Missing values come last whichever way the
   * sort goes, ties keep their original order, and text is compared by code unit, so the result
   * does not depend on the browser's locale.
   */
  function sortOrder(values, numeric, descending) {
    var order = values.map(function (_, i) { return i; });
    order.sort(function (i, j) {
      var a = values[i], b = values[j];
      var missingA = isMissing(a), missingB = isMissing(b);
      if (missingA || missingB) {
        return missingA === missingB ? i - j : (missingA ? 1 : -1);
      }
      var c = numeric ? a - b : (a < b ? -1 : (a > b ? 1 : 0));
      if (c === 0) return i - j;
      return descending ? -c : c;
    });
    return order;
  }

  /** True when every whitespace-separated term of `query` occurs in `text`, ignoring case. */
  function rowMatches(text, query) {
    var haystack = String(text).toLowerCase();
    return String(query).toLowerCase().split(/\s+/).filter(Boolean).every(function (term) {
      return haystack.indexOf(term) !== -1;
    });
  }

  function distance(ax, ay, bx, by) {
    return Math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
  }

  /** Index of the point nearest (x, y), or -1 if there is none within `limit` (if given). */
  function nearestIndex(points, x, y, limit) {
    var best = -1, bestDistance = Infinity;
    points.forEach(function (p, i) {
      var d = distance(p.x, p.y, x, y);
      if (d < bestDistance) { best = i; bestDistance = d; }
    });
    return best !== -1 && (limit === undefined || bestDistance <= limit) ? best : -1;
  }

  /** Distance from (px, py) to the segment (x1, y1)-(x2, y2). */
  function segmentDistance(px, py, x1, y1, x2, y2) {
    var dx = x2 - x1, dy = y2 - y1;
    var lengthSquared = dx * dx + dy * dy;
    if (lengthSquared === 0) return distance(px, py, x1, y1);
    var t = ((px - x1) * dx + (py - y1) * dy) / lengthSquared;
    t = Math.max(0, Math.min(1, t));
    return distance(px, py, x1 + t * dx, y1 + t * dy);
  }

  /** Distance from (x, y) to a polyline given as [{x, y}, ...] (a single point is allowed). */
  function polylineDistance(points, x, y) {
    if (points.length === 0) return Infinity;
    if (points.length === 1) return distance(points[0].x, points[0].y, x, y);
    var best = Infinity;
    for (var i = 1; i < points.length; i++) {
      var d = segmentDistance(x, y, points[i - 1].x, points[i - 1].y, points[i].x, points[i].y);
      if (d < best) best = d;
    }
    return best;
  }

  /** Index of the line nearest (x, y) within `limit`, or -1. Ties go to the later line. */
  function nearestLine(lines, x, y, limit) {
    var best = -1, bestDistance = Infinity;
    lines.forEach(function (points, i) {
      var d = polylineDistance(points, x, y);
      if (d <= bestDistance) { best = i; bestDistance = d; }
    });
    return best !== -1 && bestDistance <= limit ? best : -1;
  }

  /** Parse an SVG `points` attribute ("1,2 3,4") into [{x, y}]; malformed pairs are skipped. */
  function parsePoints(text) {
    var out = [];
    String(text).trim().split(/\s+/).forEach(function (pair) {
      var parts = pair.split(',');
      var x = parseFloat(parts[0]), y = parseFloat(parts[1]);
      if (parts.length === 2 && isFinite(x) && isFinite(y)) out.push({ x: x, y: y });
    });
    return out;
  }

  /** What a heat-map cell shows: its count, or its share of the row as a whole percent. */
  function cellText(count, share, mode) {
    return mode === 'percent' ? Math.round(share * 100) + '%' : String(count);
  }

  /** Only links to run pages are ever followed from a chart. */
  function isRunPath(path) {
    return RUN_PATH.test(String(path));
  }

  // ---- the page ------------------------------------------------------------------------------

  function element(doc, tag, className, text) {
    var node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgPoint(svg, event) {
    var matrix = svg.getScreenCTM();
    if (!matrix || !svg.createSVGPoint) return null;
    var point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    var local = point.matrixTransform(matrix.inverse());
    return { x: local.x, y: local.y };
  }

  function enhanceRunTable(doc) {
    var table = doc.querySelector('table[data-runs]');
    if (!table || !table.tBodies.length) return;
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    var headers = Array.prototype.slice.call(table.tHead.rows[0].cells);
    var metricsByRow = rows.map(function (row) {
      return parseMetrics(row.cells[row.cells.length - 1].textContent);
    });
    var names = {};
    metricsByRow.forEach(function (m) { Object.keys(m).forEach(function (n) { names[n] = true; }); });
    var metricNames = Object.keys(names).sort();

    var state = { column: -1, metric: '', descending: false, query: '' };
    var toolbar = element(doc, 'p', 'toolbar');
    var filter = element(doc, 'input');
    filter.type = 'search';
    filter.setAttribute('aria-label', 'filter runs');
    filter.placeholder = 'filter runs';
    var status = element(doc, 'span', 'muted');
    status.setAttribute('aria-live', 'polite');
    toolbar.appendChild(filter);
    var picker = null, direction = null;
    if (metricNames.length) {
      picker = element(doc, 'select');
      picker.setAttribute('aria-label', 'sort by metric');
      picker.appendChild(element(doc, 'option', '', 'sort by metric'));
      picker.firstChild.value = '';
      metricNames.forEach(function (name) {
        var option = element(doc, 'option', '', name);
        option.value = name;
        picker.appendChild(option);
      });
      direction = element(doc, 'button', '', 'high to low');
      direction.type = 'button';
      toolbar.appendChild(picker);
      toolbar.appendChild(direction);
    }
    toolbar.appendChild(status);
    table.parentNode.insertBefore(toolbar, table);

    function apply() {
      var values, numeric;
      if (state.metric) {
        values = metricsByRow.map(function (m) {
          return Object.prototype.hasOwnProperty.call(m, state.metric) ? m[state.metric] : null;
        });
        numeric = true;
      } else if (state.column >= 0) {
        values = rows.map(function (row) { return row.cells[state.column].textContent.trim(); });
        numeric = false;
      } else {
        values = rows.map(function (_, i) { return i; });
        numeric = true;
      }
      var descending = state.metric || state.column >= 0 ? state.descending : false;
      sortOrder(values, numeric, descending).forEach(function (i) { body.appendChild(rows[i]); });
      var shown = 0;
      rows.forEach(function (row) {
        var visible = rowMatches(row.textContent, state.query);
        row.hidden = !visible;
        if (visible) shown += 1;
      });
      status.textContent = shown + ' of ' + rows.length + ' runs shown';
      headers.forEach(function (th, i) {
        if (th.hasAttribute('data-sort')) {
          th.setAttribute('aria-sort', state.column === i && !state.metric
            ? (state.descending ? 'descending' : 'ascending') : 'none');
        }
      });
      if (direction) direction.textContent = state.descending ? 'high to low' : 'low to high';
    }

    headers.forEach(function (th, i) {
      if (!th.hasAttribute('data-sort')) return;
      th.tabIndex = 0;
      th.setAttribute('role', 'button');
      th.setAttribute('aria-sort', 'none');
      th.className = 'sortable';
      function toggle() {
        state.descending = state.column === i && !state.metric ? !state.descending : false;
        state.column = i;
        state.metric = '';
        if (picker) picker.value = '';
        apply();
      }
      th.addEventListener('click', toggle);
      th.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });
    });
    filter.addEventListener('input', function () { state.query = filter.value; apply(); });
    if (picker && direction) {
      picker.addEventListener('change', function () {
        state.metric = picker.value;
        state.column = -1;
        state.descending = true;
        apply();
      });
      direction.addEventListener('click', function () {
        state.descending = !state.descending;
        if (!state.metric && state.column < 0) { state.descending = false; }
        apply();
      });
    }
    apply();
  }

  function readoutFor(doc, svg) {
    var out = element(doc, 'p', 'readout', HINT);
    out.setAttribute('aria-live', 'polite');
    svg.parentNode.insertBefore(out, svg.nextSibling);
    return out;
  }

  function enhancePointCharts(doc) {
    Array.prototype.forEach.call(doc.querySelectorAll('svg[data-kind="points"]'), function (svg) {
      var dots = Array.prototype.slice.call(svg.querySelectorAll('circle[data-readout]'));
      if (!dots.length) return;
      var out = readoutFor(doc, svg);
      var points = dots.map(function (d) {
        return { x: parseFloat(d.getAttribute('cx')), y: parseFloat(d.getAttribute('cy')) };
      });
      var active = -1;
      function show(index) {
        if (index === active) return;
        if (active >= 0) dots[active].classList.remove('active');
        active = index;
        if (index >= 0) dots[index].classList.add('active');
        out.textContent = index >= 0 ? dots[index].getAttribute('data-readout') : HINT;
      }
      svg.addEventListener('pointermove', function (event) {
        var p = svgPoint(svg, event);
        show(p ? nearestIndex(points, p.x, p.y, 40) : -1);
      });
      svg.addEventListener('pointerleave', function () { show(-1); });
    });
  }

  function enhanceParallel(doc) {
    Array.prototype.forEach.call(doc.querySelectorAll('svg[data-kind="parallel"]'), function (svg) {
      var marks = Array.prototype.slice.call(svg.querySelectorAll('[data-readout]'));
      if (!marks.length) return;
      var out = readoutFor(doc, svg);
      var lines = marks.map(function (m) {
        return m.tagName.toLowerCase() === 'polyline'
          ? parsePoints(m.getAttribute('points'))
          : [{ x: parseFloat(m.getAttribute('cx')), y: parseFloat(m.getAttribute('cy')) }];
      });
      var active = -1;
      function show(index) {
        if (index === active) return;
        if (active >= 0) marks[active].classList.remove('active');
        active = index;
        svg.classList.toggle('pc-focus', index >= 0);
        if (index >= 0) marks[index].classList.add('active');
        out.textContent = index >= 0
          ? marks[index].getAttribute('data-readout') : HINT;
      }
      svg.addEventListener('pointermove', function (event) {
        var p = svgPoint(svg, event);
        show(p ? nearestLine(lines, p.x, p.y, 10) : -1);
      });
      svg.addEventListener('pointerleave', function () { show(-1); });
      svg.addEventListener('click', function () {
        var href = active >= 0 ? marks[active].getAttribute('data-href') : null;
        if (href && isRunPath(href)) doc.defaultView.location.assign(href);
      });
    });
  }

  function enhanceHeatmaps(doc) {
    Array.prototype.forEach.call(doc.querySelectorAll('svg[data-kind="heatmap"]'), function (svg) {
      var cells = Array.prototype.slice.call(svg.querySelectorAll('text[data-count]'));
      if (!cells.length) return;
      var button = element(doc, 'button', '', 'show percent of row');
      button.type = 'button';
      var mode = 'count';
      button.addEventListener('click', function () {
        mode = mode === 'count' ? 'percent' : 'count';
        cells.forEach(function (cell) {
          cell.textContent = cellText(
            parseInt(cell.getAttribute('data-count'), 10),
            parseFloat(cell.getAttribute('data-share')), mode);
        });
        button.textContent = mode === 'count' ? 'show percent of row' : 'show counts';
      });
      svg.parentNode.insertBefore(button, svg);
    });
  }

  function enhanceForms(doc) {
    Array.prototype.forEach.call(doc.querySelectorAll('form[data-autosubmit]'), function (form) {
      Array.prototype.forEach.call(form.querySelectorAll('select'), function (select) {
        select.addEventListener('change', function () { form.submit(); });
      });
    });
  }

  function init() {
    enhanceRunTable(document);
    enhancePointCharts(document);
    enhanceParallel(document);
    enhanceHeatmaps(document);
    enhanceForms(document);
  }

  return {
    parseMetrics: parseMetrics,
    sortOrder: sortOrder,
    rowMatches: rowMatches,
    nearestIndex: nearestIndex,
    segmentDistance: segmentDistance,
    polylineDistance: polylineDistance,
    nearestLine: nearestLine,
    parsePoints: parsePoints,
    cellText: cellText,
    isRunPath: isRunPath,
    init: init
  };
});
