from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

class COOPCOEPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on port {port} with COOP/COEP headers...")
    HTTPServer(('', port), COOPCOEPRequestHandler).serve_forever()
