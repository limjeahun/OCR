// OCR Recognition Worker
// This worker handles heavy ONNX inference off the main thread
// Compatible with Next.js Turbopack

// IMPORTANT: Workaround for Turbopack compatibility
// Turbopack may inject process polyfills which cause issues in Worker context
// We explicitly ensure process is undefined before loading ONNX Runtime
if (typeof process !== 'undefined') {
    self.process = undefined;
}

// Get origin from worker location
const ORIGIN = self.location.origin;

// Load ONNX Runtime
importScripts(ORIGIN + '/ort.wasm.min.js');

// Configure WASM paths after loading ort
if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = ORIGIN + '/';
    // Disable multi-threading to avoid Turbopack Worker bundling issues
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
}

let recSession = null;
let keys = [];

// Decode recognition output using CTC decoding
function decode(data, dims) {
    const timeSteps = dims[1];
    const numClasses = dims[2];
    const sb = [];
    let confSum = 0;
    let charCount = 0;
    let lastIndex = -1;

    for (let t = 0; t < timeSteps; t++) {
        let maxVal = -Infinity;
        let maxIdx = -1;
        const offset = t * numClasses;

        for (let c = 0; c < numClasses; c++) {
            if (data[offset + c] > maxVal) {
                maxVal = data[offset + c];
                maxIdx = c;
            }
        }

        // Skip blank token (index 0)
        if (maxIdx !== -1 && maxIdx !== 0) {
            // CTC: collapse repeated characters
            if (maxIdx !== lastIndex) {
                const realIdx = maxIdx - 1;
                if (realIdx >= 0 && realIdx < keys.length) {
                    sb.push(keys[realIdx]);
                    confSum += maxVal;
                    charCount++;
                }
            }
        }
        lastIndex = maxIdx;
    }

    return { text: sb.join(''), conf: charCount > 0 ? 1 : 0 };
}

// Initialize models
async function initModels() {
    // Load dictionary
    const dictResponse = await fetch(ORIGIN + '/models/korean_dict.txt');
    const dictText = await dictResponse.text();
    keys = dictText.split('\n').map(line => line.trim());
    keys.push(' ');

    // Load recognition model
    recSession = await ort.InferenceSession.create(ORIGIN + '/models/ocr_rec_kor.onnx', {
        executionProviders: ['wasm']
    });

    console.log('[OCR Worker] Recognition model loaded');
}

// Recognize single image patch
async function recognizeSingle(imageData, width, height) {
    const tensor = new ort.Tensor('float32', imageData, [1, 3, height, width]);
    const results = await recSession.run({ 'x': tensor });
    const logits = results[Object.keys(results)[0]];
    return decode(logits.data, logits.dims);
}

// Message handler
self.onmessage = async (e) => {
    const { type, id, data, batchData } = e.data;

    try {
        if (type === 'init') {
            await initModels();
            self.postMessage({ type: 'ready', id });
        } else if (type === 'recognize') {
            const result = await recognizeSingle(data.imageData, data.width, data.height);
            self.postMessage({ type: 'result', id, result });
        } else if (type === 'recognizeBatch') {
            const results = [];
            for (let i = 0; i < batchData.length; i++) {
                const item = batchData[i];
                const result = await recognizeSingle(item.imageData, item.width, item.height);
                results.push(result);
                self.postMessage({
                    type: 'progress',
                    id,
                    progress: 'box_' + (i + 1) + '_of_' + batchData.length
                });
            }
            self.postMessage({ type: 'result', id, batchResults: results });
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            id,
            error: error.message || String(error)
        });
    }
};
