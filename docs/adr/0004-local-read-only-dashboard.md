# 4. A local, read-only dashboard rendered on the server

Status: accepted; the "no JavaScript" consequence is revised by [ADR 9](0009-a-small-same-origin-script-for-the-dashboard.md)

## Context
A web view is the easiest way to look at many runs, but a local server is still a network
service that a web page in another tab could try to reach.

## Decision
The dashboard only reads. It binds to 127.0.0.1 by default, rejects requests whose `Host` header
is not an allowed loopback name (DNS rebinding), sends `Content-Security-Policy: default-src
'none'` with inline styles only, escapes all text, and ships charts as server-side SVG. On
Windows it takes the port exclusively so another process cannot share it.

## Consequences
- Without a script the pages offer no client-side interactivity beyond links and forms.
- Listening on another interface requires `--allow-network` and is the user's responsibility.
