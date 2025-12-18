// OCR Recognition Worker
// This worker handles heavy ONNX inference off the main thread

import * as ort from 'onnxruntime-web';

// Configure WASM paths for worker context
ort.env.wasm.wasmPaths = '/';

interface RecognitionRequest {
    type: 'init' | 'recognize' | 'recognizeBatch';
    id: string;
    data?: {
        imageData: Float32Array;
        width: number;
        height: number;
    };
    batchData?: {
        imageData: Float32Array;
        width: number;
        height: number;
    }[];
}

interface RecognitionResponse {
    type: 'progress' | 'result' | 'error' | 'ready';
    id: string;
    progress?: string;
    result?: {
        text: string;
        conf: number;
    };
    batchResults?: {
        text: string;
        conf: number;
    }[];
    error?: string;
}

let recSession: ort.InferenceSession | null = null;
let keys: string[] = [];

// Decode recognition output
function decode(data: Float32Array, dims: readonly number[], keys: string[]): { text: string, conf: number } {
    const timeSteps = dims[1];
    const numClasses = dims[2];

    const sb: string[] = [];
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

        const blankIdx = 0;

        if (maxIdx !== -1 && maxIdx !== blankIdx) {
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

    return {
        text: sb.join(''),
        conf: charCount > 0 ? 1 : 0
    };
}

// Initialize models
async function initModels(): Promise<void> {
    try {
        // Load dictionary
        const dictResponse = await fetch('/models/korean_dict.txt');
        const dictText = await dictResponse.text();
        keys = dictText.split('\n').map(line => line.trim());
        keys.push(' ');

        // Load recognition model
        const option: ort.InferenceSession.SessionOptions = {
            executionProviders: ['wasm'],
        };

        recSession = await ort.InferenceSession.create('/models/ocr_rec_kor.onnx', option);

        console.log('[OCR Worker] Recognition model loaded');
    } catch (e) {
        console.error('[OCR Worker] Failed to load models:', e);
        throw e;
    }
}

// Recognize single image
async function recognizeSingle(imageData: Float32Array, width: number, height: number): Promise<{ text: string, conf: number }> {
    if (!recSession) throw new Error('Model not initialized');

    const recTensor = new ort.Tensor('float32', imageData, [1, 3, height, width]);
    const recResults = await recSession.run({ 'x': recTensor });
    const recLogits = recResults[Object.keys(recResults)[0]];

    return decode(recLogits.data as Float32Array, recLogits.dims, keys);
}

// Message handler
self.onmessage = async (event: MessageEvent<RecognitionRequest>) => {
    const { type, id, data, batchData } = event.data;

    try {
        switch (type) {
            case 'init':
                await initModels();
                self.postMessage({ type: 'ready', id } as RecognitionResponse);
                break;

            case 'recognize':
                if (!data) throw new Error('No image data provided');
                const result = await recognizeSingle(data.imageData, data.width, data.height);
                self.postMessage({ type: 'result', id, result } as RecognitionResponse);
                break;

            case 'recognizeBatch':
                if (!batchData || batchData.length === 0) throw new Error('No batch data provided');
                const batchResults: { text: string; conf: number }[] = [];

                for (let i = 0; i < batchData.length; i++) {
                    const item = batchData[i];
                    const itemResult = await recognizeSingle(item.imageData, item.width, item.height);
                    batchResults.push(itemResult);

                    // Send progress update
                    self.postMessage({
                        type: 'progress',
                        id,
                        progress: `Processing ${i + 1}/${batchData.length}`
                    } as RecognitionResponse);
                }

                self.postMessage({ type: 'result', id, batchResults } as RecognitionResponse);
                break;
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            id,
            error: error instanceof Error ? error.message : String(error)
        } as RecognitionResponse);
    }
};

// Inform main thread that worker is ready
self.postMessage({ type: 'ready', id: 'startup' } as RecognitionResponse);
