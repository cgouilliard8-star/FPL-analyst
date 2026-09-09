import pathlib, re
src = pathlib.Path("/root/work/site/index.html").read_text()
data = pathlib.Path("/root/work/site/data/live.json").read_text()
body = src.split("<head>", 1)[1]
body = body.replace("</head>\n<body>", "").replace("</body>\n</html>", "").replace("</head>", "").replace("<body>", "")
body = re.sub(r"<meta[^>]*>\n?", "", body)
body = body.replace('<div class="wrap">', '<script>const EMBEDDED = ' + data + ';</script>\n<div class="wrap">', 1)
old = 'try { const r = await fetch("data/live.json", {cache:"no-cache"}); if (!r.ok) throw new Error("HTTP "+r.status); fresh = await r.json();'
assert old in body
body = body.replace(old, 'try { fresh = EMBEDDED;')
pathlib.Path("/root/fpl-dashboard.html").write_text(body.strip() + "\n")
