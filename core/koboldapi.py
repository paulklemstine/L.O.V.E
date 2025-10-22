import threading
import html
import re
import time
import requests

class Controller:
    def __init__(self):
        # Define HTML-Tag-Clearing Regex
        self.cleaner = re.compile('<.*?>')

        # Settings
        self.debug = False
        self.url = ""
        self.sid = ""
        self.reset_after_input = False

        # Runtime Variables
        self.chunks = []
        self.inputs = 0
        self.ready = False
        self.addSpaceBefore = False
        self.gameStarted = False

    def ResetStory(self):
        self.chunks = []
        self.inputs = 0
        self.addSpaceBefore = False
        self.ready = False

        r = requests.post(self.url, data='42["message",{"cmd":"newgame","data":""}]')

    def SetMemory(self, memory):
        r = requests.post(self.url, data='42["message",{"cmd":"memory","data":""}]')
        r = requests.post(self.url, data='42["message",{"cmd":"submit","actionmode":0,"data":"' + memory.replace('"', '\\"').replace("\n", "\\n") + '"}]')

    def Retry(self):
        self.chunks.pop()
        self.chunks.pop()
        self.inputs = self.inputs - 1
        r = requests.post(self.url, data='42["message",{"cmd":"retry","data":""}]')
        return self.GetOutput()

    def CommandParser(self):
        if self.debug:
            print("KOBOLDAPI DEBUG: Started Output Loop Thread Successfully!")

        while True:
            r = requests.get(self.url) # Request Data
            r.encoding = 'utf-8'
            output = str(r.content).encode('utf-8').decode('unicode-escape').encode('utf-16', 'surrogatepass').decode('utf-16') # Get Output
            if not output == "b'2'": # Ignore Keep-Alive Acknowledgement Outputs
                if self.debug:
                    print("KOBOLDAPI DEBUG: Received Initial Output: '" + output + "'")
                if '"gamestarted":true' in output:
                    self.gameStarted = True
                    if self.debug:
                        print("KOBOLDAPI DEBUG: Game Started!")
                if '"gamestarted":false' in output:
                    self.gameStarted = False
                    if self.debug:
                        print("KOBOLDAPI DEBUG: Game Ended!")
                while 'cmd' in output:
                    output = output[output.index('"cmd":"')+7:]
                    cmd = output[:output.index('"')]
                    if self.debug:
                        print("KOBOLDAPI DEBUG: Received Command: '" + cmd + "'")

                    if (cmd == 'updatescreen' or cmd == 'updatechunk') and not 'generating story' in output and self.gameStarted and '"data":"ready"' in output:
                        while '<chunk' in output:
                            output = output[output.index('<chunk')+6:]
                            output = output[output.index('>')+1:]
                            chunk = output[:output.index('</chunk>')]
                            chunk = html.unescape(chunk)
                            chunk = chunk.replace('<br/>', '\n')
                            self.chunks.append(chunk)
                        self.ready = True
                        if self.debug:
                            print("KOBOLDAPI DEBUG: Ready with New Chunks!")
            r = requests.post(self.url, data="3") # Keep-Alive Request

    def Initialise(self, _url, _debug=False, _reset_after_input=False):
        self.url = _url
        self.debug = _debug
        self.reset_after_input = _reset_after_input

        if self.debug:
            print("KOBOLDAPI DEBUG: Debug Mode is Enabled!")

        if not self.url.startswith("https://") and not self.url.startswith("http://"):
            self.url = "https://" + self.url

        try:
            if requests.get(self.url).status_code != 200:
                print("KOBOLDAPI ERROR: URL is not Reachable! Halting...")
                return False
        except:
            print("KOBOLDAPI ERROR: URL is not Reachable! Halting...")
            return False

        if self.debug:
            print("KOBOLDAPI DEBUG: KoboldAPI Ready!")

        if self.url.endswith("#"):
            self.url = self.url[:-1]
        if not self.url.endswith("/"):
            self.url = self.url + "/"
        self.url = self.url + "socket.io/?EIO=4&transport=polling&t=0"

        r = requests.get(self.url)

        self.sid = str(r.content)
        self.sid = self.sid[self.sid.index('"sid":')+7:]
        self.sid = self.sid[:self.sid.index('"')]

        self.url = self.url + "&sid=" + self.sid

        r = requests.post(self.url, data="40")

        self.ResetStory()

        t = threading.Thread(target=self.CommandParser)
        t.daemon = True
        t.start()

        return True

    def Generate(self, textin, new_only=False):
        if self.addSpaceBefore:
            textin = " " + textin

        gen_cmd = '42["message",{"cmd":"submit","actionmode":0,"data":"' + textin.replace('"', '\\"').replace("\n", "\\n") + '"}]'
        if self.debug:
            print("KOBOLDAPI DEBUG: URL: " + self.url + " Payload: " + gen_cmd)

        r = requests.post(self.url, data=gen_cmd.encode('utf-8'), headers={'Content-type': 'text/plain; charset=utf-8'})

        if len(textin) > 0 and len(self.chunks) > 0:
            self.chunks.append(textin.replace("\\n", "\n"))

        return self.GetOutput(textin, new_only)

    def GetOutput(self, textin="", new_only=False):
        output = ""
        while True:
            if self.ready == True:
                for chunk in self.chunks:
                    output = output + chunk
                self.ready = False
                break

        output = output.encode('utf-8').decode('unicode-escape').encode('utf-16', 'surrogatepass').decode('utf-16')

        if self.reset_after_input:
            self.ResetStory()

        if new_only:
            output = textin.replace("\\n", "\n") + self.chunks[len(self.chunks)]

        if not output.endswith("\n") and not output.endswith(" "):
            self.addSpaceBefore = True

        self.inputs = self.inputs + 1

        return output
