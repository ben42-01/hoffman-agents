const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5000;
const SITE_DIR = path.join(__dirname, 'site');

const MIME = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.md': 'text/markdown',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

// Resolve a request to a file path. Checks site/ first, then the Node and
// Python lib roots (for examples, docs, etc.), ensuring directory traversal safety.
const LIB_DIRS = [
  SITE_DIR,
  path.join(__dirname, 'hoffman-agents-node'),
  path.join(__dirname, 'hoffman-agents-python'),
];

function resolvePath(url) {
  if (url === '/') url = '/index.html';
  for (const base of LIB_DIRS) {
    const p = path.join(base, url);
    if (p.startsWith(base) && fs.existsSync(p)) return p;
  }
  return null;
}

const server = http.createServer((req, res) => {
  let url = req.url.split('?')[0];
  const filePath = resolvePath(url);

  if (!filePath) {
    res.writeHead(404);
    return res.end('Not Found');
  }

  const ext = path.extname(filePath);
  const contentType = MIME[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(500);
      return res.end('Internal Server Error');
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`  http://localhost:${PORT}`);
});
