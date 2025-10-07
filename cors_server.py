import http.server
import socketserver

class COOPCOEPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Access-Control-Allow-Origin', '*')
        return super().end_headers()

PORT = 8000

with socketserver.TCPServer(("", PORT), COOPCOEPRequestHandler) as httpd:
    print(f"Serving at port {PORT} with COOP/COEP headers")
    httpd.serve_forever()