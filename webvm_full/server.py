from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
import os
from email.utils import formatdate
import hashlib

class COOPCOEPRangeRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()
    
    def get_etag(self, path):
        """Generate ETag from file path and modification time"""
        stat = os.stat(path)
        etag_data = f"{path}-{stat.st_mtime}-{stat.st_size}"
        return hashlib.md5(etag_data.encode()).hexdigest()
    
    def do_HEAD(self):
        """Handle HEAD requests - CheerpX uses this to check range support"""
        path = self.translate_path(self.path)
        
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return
            
        if os.path.isdir(path):
            return super().do_HEAD()
        
        file_size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        
        self.send_response(200)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Length', str(file_size))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Last-Modified', formatdate(mtime, usegmt=True))
        self.send_header('ETag', f'"{self.get_etag(path)}"')
        self.end_headers()
    
    def do_GET(self):
        # Handle range requests for HttpBytesDevice
        path = self.translate_path(self.path)
        
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return
            
        if os.path.isdir(path):
            return super().do_GET()
        
        file_size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        
        # Check if this is a range request
        range_header = self.headers.get('Range')
        
        if range_header:
            # Parse range header (format: bytes=start-end)
            try:
                range_spec = range_header.replace('bytes=', '')
                range_parts = range_spec.split('-')
                start = int(range_parts[0]) if range_parts[0] else 0
                end = int(range_parts[1]) if range_parts[1] else file_size - 1
                
                # Validate range
                if start >= file_size or end >= file_size or start > end:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return
                
                # Send partial content response
                self.send_response(206)
                self.send_header('Content-Type', self.guess_type(path))
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(end - start + 1))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Last-Modified', formatdate(mtime, usegmt=True))
                self.send_header('ETag', f'"{self.get_etag(path)}"')
                self.end_headers()
                
                # Send the requested byte range
                with open(path, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(end - start + 1))
            except Exception as e:
                print(f"Error handling range request: {e}")
                self.send_error(500, "Internal Server Error")
        else:
            # Normal request - send full file with range support headers
            self.send_response(200)
            self.send_header('Content-Type', self.guess_type(path))
            self.send_header('Content-Length', str(file_size))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Last-Modified', formatdate(mtime, usegmt=True))
            self.send_header('ETag', f'"{self.get_etag(path)}"')
            self.end_headers()
            
            with open(path, 'rb') as f:
                self.wfile.write(f.read())

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on port {port} with COOP/COEP headers and range request support...")
    HTTPServer(('', port), COOPCOEPRangeRequestHandler).serve_forever()
