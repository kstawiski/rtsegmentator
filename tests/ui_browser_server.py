"""Serve synthetic UI browser checks without starting the backend or running models.

Open /__qa, then choose Run checks. Requires only the Python standard library.
"""
from argparse import ArgumentParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/__qa":
            body = '''<!doctype html><meta charset="utf-8"><title>Synthetic UI checks</title>
<style>body{margin:0;font:14px system-ui}iframe{display:block;width:1280px;height:800px;border:0}#result{white-space:pre-wrap}</style>
<button id="run">Run checks</button><span>Synthetic UI fixtures; no patient images or inference.</span>
<iframe id="page" src="/__ui_fixture"></iframe><pre id="result"></pre>
<script src="/tests/ui_browser_checks.js"></script><script>
document.getElementById('run').onclick=async()=>{
  const output=document.getElementById('result');output.textContent='Running…';
  const frame=document.getElementById('page');
  await new Promise(resolve=>{frame.onload=resolve;frame.src='/__ui_fixture';});
  try{output.textContent=JSON.stringify(await runUiBrowserChecks(frame),null,2);}
  catch(error){output.textContent=error.stack;}
};</script>'''
        elif self.path == "/__ui_fixture":
            body = (ROOT / "static/index.html").read_text().replace(
                "<head>", '<head><script src="/tests/ui_fixture.js"></script>'
            )
        elif self.path.split("?")[0] in {
            "/static/index.html", "/static/app.js", "/static/app.css",
            "/static/viewer.js", "/static/ui-catalog.json",
            "/tests/ui_fixture.js", "/tests/ui_browser_checks.js",
        }:
            return super().do_GET()
        else:
            self.send_error(404)
            return
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18743)
    args = parser.parse_args()
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()
