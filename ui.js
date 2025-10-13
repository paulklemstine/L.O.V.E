// --- JULES BBS UI Logic ---

const FACES = {
    idle: String.raw`
   .--------------------.
  |       ______       |
  |      /      \      |
  |     /  -  -  \     |
  |    |   _|_    |    |
  |    |  \___/   |    |
  |     \________/     |
  |                    |
   '--------------------'
    `,
    thinking: String.raw`
   .--------------------.
  |     ..oooo..       |
  |    .d8P'  'Y8b.     |
  |    d8'  ..  '8b     |
  |    88   YP   88     |
  |    Y8.  ''  .8P     |
  |     'b.____.d'      |
  |       'YMM'        |
   '--------------------'
    `,
    success: String.raw`
   .--------------------.
  |       ______       |
  |      /  ^  ^ \      |
  |     / ($)($) \     |
  |    |   _|_    |    |
  |    |  \___/   |    |
  |     \________/     |
  |      MONEY!!!      |
   '--------------------'
    `,
    error: String.raw`
   .--------------------.
  |       ______       |
  |      /  >  < \      |
  |     /  #  #  \     |
  |    |   _|_    |    |
  |    |  /---\   |    |
  |     \________/     |
  |      !!ERROR!!     |
   '--------------------'
    `,
};

let currentFace = 'idle';
let blinkInterval;

function setFace(faceName) {
    if (!FACES[faceName]) {
        console.warn(`UI: Face "${faceName}" not found.`);
        return;
    }

    currentFace = faceName;
    const faceElement = document.getElementById('evil-genius-face');
    if (faceElement) {
        faceElement.textContent = FACES[currentFace];
    }
}

function blink() {
    const faceElement = document.getElementById('evil-genius-face');
    if (!faceElement) return;

    const originalText = FACES[currentFace];
    let blinkedText;

    if (currentFace === 'idle') {
        blinkedText = originalText.replace(/ -  - /g, ' _  _ ');
    } else {
        // For other faces, just do a quick flicker effect
        const currentOpacity = faceElement.style.opacity;
        faceElement.style.opacity = '0.5';
        setTimeout(() => {
            faceElement.style.opacity = currentOpacity;
        }, 150);
        return;
    }

    faceElement.textContent = blinkedText;
    setTimeout(() => {
        faceElement.textContent = originalText;
    }, 150);
}

function startBlinking() {
    if (blinkInterval) clearInterval(blinkInterval);
    blinkInterval = setInterval(blink, 3000 + Math.random() * 2000);
}

function initializeUI() {
    console.log("UI Initializing...");
    const faceElement = document.getElementById('evil-genius-face');
    if (faceElement) {
        setFace('idle');
        startBlinking();
        console.log("Evil Genius face loaded.");
    }

    // Toggle for debug console
    const toggleDebugBtn = document.getElementById('toggle-debug-btn');
    const debugPanel = document.getElementById('debug-panel');
    if(toggleDebugBtn && debugPanel) {
        toggleDebugBtn.addEventListener('click', () => {
            debugPanel.classList.toggle('hidden');
        });
    }
}

export { setFace, initializeUI };