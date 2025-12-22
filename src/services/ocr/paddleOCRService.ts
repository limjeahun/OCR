import * as ort from 'onnxruntime-web';
import { OCRResult } from './types';
import { openCVService } from './opencvService';

// Configure WASM paths to use files from public/ folder
ort.env.wasm.wasmPaths = '/';
// Disable multi-threading to avoid Turbopack Worker bundling issues
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

interface WorkerMessage {
    type: 'progress' | 'result' | 'error' | 'ready';
    id: string;
    progress?: string;
    result?: { text: string; conf: number };
    batchResults?: { text: string; conf: number }[];
    error?: string;
}

export class PaddleOCRService {
    private detSession: ort.InferenceSession | null = null;
    private recSession: ort.InferenceSession | null = null;
    private keys: string[] = [];
    private isInitializing = false;

    // Worker for recognition (offloads heavy inference)
    private worker: Worker | null = null;
    private workerReady = false;
    private pendingRequests = new Map<string, {
        resolve: (value: any) => void;
        reject: (error: any) => void;
        onProgress?: (status: string) => void;
    }>();
    private requestId = 0;

    private initWorker(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.workerReady && this.worker) {
                resolve();
                return;
            }

            try {
                // Use separate Worker file for Next.js Turbopack compatibility
                // Worker file is located in public/workers/
                this.worker = new Worker('/workers/ocrRecognitionWorker.js');

                this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
                    const { type, id, progress, result, batchResults, error } = event.data;

                    if (type === 'ready' && id === 'init') {
                        this.workerReady = true;
                        const pending = this.pendingRequests.get(id);
                        if (pending) {
                            pending.resolve(undefined);
                            this.pendingRequests.delete(id);
                        }
                        resolve();
                        return;
                    }

                    const pending = this.pendingRequests.get(id);
                    if (!pending) return;

                    switch (type) {
                        case 'progress':
                            pending.onProgress?.(progress || '');
                            break;
                        case 'result':
                            pending.resolve(batchResults || result);
                            this.pendingRequests.delete(id);
                            break;
                        case 'error':
                            pending.reject(new Error(error));
                            this.pendingRequests.delete(id);
                            break;
                    }
                };

                this.worker.onerror = (error) => {
                    console.error('[PaddleOCR] Worker error:', error);
                    reject(error);
                };

                // Initialize worker
                const initId = 'init';
                this.pendingRequests.set(initId, { resolve, reject });
                this.worker.postMessage({ type: 'init', id: initId });

            } catch (e) {
                console.error('[PaddleOCR] Failed to create worker:', e);
                reject(e);
            }
        });
    }

    private sendToWorker(type: string, data: any, onProgress?: (status: string) => void): Promise<any> {
        return new Promise((resolve, reject) => {
            if (!this.worker) {
                reject(new Error('Worker not initialized'));
                return;
            }

            const id = `req_${this.requestId++}`;
            this.pendingRequests.set(id, { resolve, reject, onProgress });
            this.worker.postMessage({ type, id, ...data });
        });
    }

    async init(onProgress?: (status: string) => void) {
        if (this.detSession && this.workerReady) return;
        if (this.isInitializing) {
            while (this.isInitializing) {
                await new Promise(r => setTimeout(r, 100));
            }
            return;
        }
        this.isInitializing = true;

        try {
            onProgress?.('Loading OCR Dictionary...');
            // Load Dictionary (needed for main thread field detection)
            if (this.keys.length === 0) {
                const response = await fetch('/models/korean_dict.txt');
                const text = await response.text();
                this.keys = text.split('\n').map(line => line.trim());
                this.keys.push(' ');
            }

            const option: ort.InferenceSession.SessionOptions = {
                // PaddleOCR model uses ceil_mode in MaxPool operation
                // which is NOT supported by WebGPU or WebGL in onnxruntime-web
                // Using WASM (CPU) for maximum compatibility
                executionProviders: ['wasm'],
            };

            // Load Detection Model (stays on main thread - runs only once per image)
            if (!this.detSession) {
                onProgress?.('Loading Detection Model (88MB)...');
                this.detSession = await ort.InferenceSession.create('/models/ocr_det.onnx', option);
            }

            // Initialize Worker for Recognition (runs N times per image)
            onProgress?.('Initializing Recognition Worker...');
            await this.initWorker();

            console.log('PaddleOCR models loaded (Detection: main, Recognition: worker)');
        } catch (e) {
            console.error('Failed to load PaddleOCR models', e);
            throw e;
        } finally {
            this.isInitializing = false;
        }
    }

    async recognize(
        image: HTMLImageElement | HTMLCanvasElement,
        onProgress?: (status: string) => void,
        documentType?: string
    ): Promise<OCRResult> {
        // Ensure models are loaded
        if (!this.detSession || !this.workerReady) await this.init(onProgress);
        if (!this.detSession || !this.worker) throw new Error('Models failed to load');

        const cv = window.cv;
        let src = cv.imread(image);

        try {
            if (src.channels() === 4) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
            }

            // ---------------------------
            // Dynamic Detection Threshold based on Document Type
            // ---------------------------
            // BUSINESS_REGISTRATION: Slightly higher threshold (was 0.45, reduced to 0.35 for better accuracy)
            // ID_CARD/DRIVER_LICENSE: Standard threshold
            // Default: Conservative threshold
            const getDetectionThreshold = (docType?: string): number => {
                switch (docType) {
                    case 'BUSINESS_REGISTRATION':
                        return 0.35; // Reduced from 0.45 - was too aggressive
                    case 'ID_CARD':
                        return 0.33;
                    case 'DRIVER_LICENSE':
                        return 0.33;
                    default:
                        return 0.3;  // Default
                }
            };
            const boxThreshold = getDetectionThreshold(documentType);
            console.log(`[PaddleOCR] Using detection threshold: ${boxThreshold} for ${documentType || 'unknown'}`);

            // ---------------------------
            // 1. Detection (Main Thread - runs once)
            // ---------------------------
            onProgress?.('Detecting Text Regions...');

            const limitSide = 1280;
            let ratio = 1.0;
            let w = src.cols;
            let h = src.rows;

            if (Math.max(h, w) > limitSide) {
                if (h > w) ratio = limitSide / h;
                else ratio = limitSide / w;
            }

            let resizeH = Math.round(h * ratio);
            let resizeW = Math.round(w * ratio);

            resizeH = Math.round(resizeH / 32) * 32;
            resizeW = Math.round(resizeW / 32) * 32;

            let detInput = new cv.Mat();
            cv.resize(src, detInput, new cv.Size(resizeW, resizeH));

            const detData = openCVService.blobFromImage(detInput, 1 / 255.0, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
            detInput.delete();

            const detTensor = new ort.Tensor('float32', detData, [1, 3, resizeH, resizeW]);
            const detResults = await this.detSession.run({ 'x': detTensor });
            const mapData = detResults[Object.keys(detResults)[0]].data as Float32Array;

            const boxes = openCVService.getBoxesFromMap(mapData, resizeW, resizeH, boxThreshold, documentType);

            const scaleX = w / resizeW;
            const scaleY = h / resizeH;

            boxes.forEach(box => {
                box.points.forEach((p: any) => {
                    p.x *= scaleX;
                    p.y *= scaleY;
                });
            });

            boxes.sort((a, b) => a.center.y - b.center.y);

            // ---------------------------
            // 2. Group boxes into lines
            // ---------------------------
            const lines: any[][] = [];
            let currentLine: any[] = [];

            for (const box of boxes) {
                if (currentLine.length === 0) {
                    currentLine.push(box);
                    continue;
                }
                const firstBox = currentLine[0];
                const yDiff = Math.abs(box.center.y - firstBox.center.y);
                const height = Math.min(box.size.height, firstBox.size.height);

                // Tightened from 0.3 to 0.15 to prevent merging of adjacent rows
                if (yDiff < height * 0.15) {
                    currentLine.push(box);
                } else {
                    currentLine.sort((a, b) => a.center.x - b.center.x);
                    lines.push(currentLine);
                    currentLine = [box];
                }
            }
            if (currentLine.length > 0) {
                currentLine.sort((a, b) => a.center.x - b.center.x);
                lines.push(currentLine);
            }

            // ---------------------------
            // 3. Prepare batch data for worker (off-thread recognition)
            // ---------------------------
            onProgress?.(`Preparing ${boxes.length} text regions...`);
            await new Promise(r => setTimeout(r, 0)); // Yield to UI

            const allBoxes: { box: any; imageData: Float32Array; width: number; height: number }[] = [];

            for (const line of lines) {
                for (const box of line) {
                    const boxW = Math.max(box.size.width, box.size.height);
                    const boxH = Math.min(box.size.width, box.size.height);

                    let patch = openCVService.warpBox(src, box.points, Math.round(boxW), Math.round(boxH));

                    const recH = 48;
                    let recW = Math.round(recH * (boxW / boxH));

                    let recInput = new cv.Mat();
                    cv.resize(patch, recInput, new cv.Size(recW, recH), 0, 0, cv.INTER_CUBIC);

                    const recData = openCVService.blobFromImage(recInput, 1 / 255.0, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);

                    patch.delete();
                    recInput.delete();

                    allBoxes.push({
                        box,
                        imageData: recData,
                        width: recW,
                        height: recH
                    });
                }

                // Yield to UI after each line's preprocessing
                await new Promise(r => setTimeout(r, 0));
                onProgress?.(`Preparing regions (${allBoxes.length}/${boxes.length})...`);
            }

            // ---------------------------
            // 4. Recognition in Worker (non-blocking!)
            // ---------------------------
            onProgress?.(`Recognizing ${allBoxes.length} text boxes (Worker)...`);

            let processedCount = 0;
            const workerOnProgress = (status: string) => {
                if (status.startsWith('box_')) {
                    const match = status.match(/box_(\d+)_of_(\d+)/);
                    if (match) {
                        processedCount = parseInt(match[1]);
                        onProgress?.(`Recognizing (${processedCount}/${allBoxes.length})...`);
                    }
                }
            };

            const batchData = allBoxes.map(b => ({
                imageData: b.imageData,
                width: b.width,
                height: b.height
            }));

            const results = await this.sendToWorker('recognizeBatch', { batchData }, workerOnProgress) as { text: string; conf: number }[];

            // ---------------------------
            // 5. Assemble results
            // ---------------------------
            onProgress?.('Assembling results...');

            let fullText = '';
            let totalConf = 0;
            let count = 0;
            let resultIdx = 0;

            for (const line of lines) {
                let lineText = '';
                let lastBoxMaxX = -1;
                let lastBoxHeight = 0;

                for (const box of line) {
                    const decoded = results[resultIdx++];

                    if (decoded.conf > 0.5) {
                        const currentMinX = box.center.x - (box.size.width / 2);
                        if (lineText.length > 0) {
                            const gap = currentMinX - lastBoxMaxX;

                            // Extended field keywords list for better detection
                            const fieldKeywords = [
                                '법인등록번호', '법인들록번호', '번인등록번호',
                                '본점소재지', '본정소재지', '본정소재',
                                '사업장소재지', '사업장소재', '사업장',
                                '개업연월일', '개업연', '등록번호',
                                '대표자', '법인명', '단체명'
                            ];
                            const isFieldStart = fieldKeywords.some(kw =>
                                decoded.text.includes(kw) ||
                                decoded.text.replace(/\s/g, '').startsWith(kw.substring(0, 2))
                            );

                            // More aggressive newline insertion for field separation
                            // Reduced gap threshold to 0.2 (was 0.3) for field keywords
                            if (isFieldStart && gap > (lastBoxHeight * 0.2)) {
                                lineText += '\n';
                            } else if (gap > (lastBoxHeight * 0.4)) {
                                // Reduced from 0.5 to 0.4 for general large gaps
                                lineText += '\n';
                            } else if (gap > (lastBoxHeight * 0.15)) {
                                lineText += ' ';
                            }
                        }
                        lineText += decoded.text;
                        lastBoxMaxX = box.center.x + (box.size.width / 2);
                        lastBoxHeight = box.size.height;
                        totalConf += decoded.conf;
                        count++;
                    }
                }
                if (lineText.length > 0) {
                    fullText += lineText + '\n';
                }
            }

            src.delete();
            return {
                text: fullText,
                confidence: count > 0 ? totalConf / count : 0
            };

        } catch (e) {
            src.delete();
            console.error('PaddleOCR Recognition failed', e);
            throw e;
        }
    }

    // Cleanup
    dispose() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
            this.workerReady = false;
        }
    }
}

export const paddleOCRService = new PaddleOCRService();
