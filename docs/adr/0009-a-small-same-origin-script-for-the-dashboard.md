# 9. A small same-origin script for the dashboard

Status: accepted

## Context
ADR 4 kept the dashboard free of JavaScript. That is the safest choice, but a list of hundreds of
runs is hard to use without sorting and filtering, and a chart that cannot say which run a dot
belongs to is only half useful. Server-rendered forms can sort, but only by reloading the page.

## Decision
- Pages stay complete without a script: every table, chart and value is rendered on the server,
  and the script only adds behaviour. With scripts blocked every page still reads correctly.
- There is one script, `static/dashboard.js`, written in plain ES5 with no dependencies and no
  build step, shipped in the package and served by the dashboard at a single fixed path.
- The Content-Security-Policy changes from `default-src 'none'` to also allow `script-src 'self'`.
  Inline scripts, `eval`, remote hosts and `data:` stay forbidden.
- The script never builds HTML from text (no `innerHTML`), makes no requests, uses no storage or
  cookies, and follows a link only if it matches `/run/<id>`. A test scans the source for each of
  these and fails if one appears.
- Charts carry their values in `data-` attributes written by the server and escaped there, so the
  script reads attributes and sets `textContent`.
- The logic that can be separated from the page (sorting, filtering, nearest-point search, parsing)
  is plain functions with hand-computed tests under Node. Behaviour on a real page is tested in
  Chromium through Playwright when it is installed, and those tests are skipped otherwise, because
  the package itself has no dependencies.

## Consequences
- A bug in the script can break sorting or hover read-outs but not what the server renders.
- Playwright is an optional development tool (`pip install -e ".[browser]"` then
  `playwright install chromium`). The main CI matrix runs the Node tests and the string-level
  checks; a separate CI job installs Playwright and runs the browser tests with
  `MODELBENCH_REQUIRE_BROWSER_TESTS=1`, so a missing browser fails that job instead of skipping.
- Adding a second script or any third-party code would need a new decision.
