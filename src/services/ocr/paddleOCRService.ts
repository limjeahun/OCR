import * as ort from 'onnxruntime-web';
import { OCRResult } from './types';
import { openCVService } from './opencvService';

// Configure WASM paths to use files from public/ folder
ort.env.wasm.wasmPaths = '/';

export class PaddleOCRService {
    private detSession: ort.InferenceSession | null = null;
    private recSession: ort.InferenceSession | null = null;
    private keys: string[] = [];
    private isInitializing = false;

    async init(onProgress?: (status: string) => void) {
        if (this.detSession && this.recSession) return;
        if (this.isInitializing) {
            while (this.isInitializing) {
                await new Promise(r => setTimeout(r, 100));
            }
            return;
        }
        this.isInitializing = true;

        try {
            onProgress?.('Loading OCR Dictionary...');
            // Load Dictionary
            if (this.keys.length === 0) {
                const response = await fetch('/models/korean_dict.txt');
                const text = await response.text();
                this.keys = text.split('\n').map(line => line.trim());
                this.keys.push(' ');
            }

            const option: ort.InferenceSession.SessionOptions = {
                executionProviders: ['wasm'],
            };

            // Load Models - Standard Order: Detection First (Large) -> Recognition Second (Small)
            // This ensures we allocate the biggest chunk of memory first.
            if (!this.detSession) {
                onProgress?.('Loading Detection Model (88MB)...');
                this.detSession = await ort.InferenceSession.create('/models/ocr_det.onnx', option);
            }

            if (!this.recSession) {
                onProgress?.('Loading Recognition Model (13MB)...');
                this.recSession = await ort.InferenceSession.create('/models/ocr_rec_kor.onnx', option);
            }

            console.log('PaddleOCR models loaded');
        } catch (e) {
            console.error('Failed to load PaddleOCR models', e);
            throw e;
        } finally {
            this.isInitializing = false;
        }
    }

    async recognize(image: HTMLImageElement | HTMLCanvasElement, onProgress?: (status: string) => void): Promise<OCRResult> {
        // Ensure models are loaded
        if (!this.detSession || !this.recSession) await this.init(onProgress);
        if (!this.detSession || !this.recSession) throw new Error('Models failed to load');

        const cv = window.cv;
        let src = cv.imread(image);

        try {
            if (src.channels() === 4) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
            }

            // ---------------------------
            // 1. Detection
            // ---------------------------
            onProgress?.('Detecting Text Regions...');

            const limitSide = 1280; // Increased from 960 for better small text accuracy
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

            const boxes = openCVService.getBoxesFromMap(mapData, resizeW, resizeH, 0.3);

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
            // 2. Recognition
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

                if (yDiff < height * 0.3) {
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

            let fullText = '';
            let totalConf = 0;
            let count = 0;

            onProgress?.(`Recognizing ${lines.length} lines...`);
            let processedLines = 0;

            for (const line of lines) {
                processedLines++;
                if (processedLines % 5 === 0) {
                    onProgress?.(`Recognizing lines (${processedLines}/${lines.length})...`);
                    await new Promise(r => setTimeout(r, 0));
                }

                let lineText = '';
                let lastBoxMaxX = -1;
                let lastBoxHeight = 0;

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

                    const recTensor = new ort.Tensor('float32', recData, [1, 3, recH, recW]);
                    const recResults = await this.recSession.run({ 'x': recTensor });
                    const recLogits = recResults[Object.keys(recResults)[0]];

                    const decoded = this.decode(recLogits.data as Float32Array, recLogits.dims);

                    if (decoded.conf > 0.5) {
                        const currentMinX = box.center.x - (box.size.width / 2);
                        if (lineText.length > 0) {
                            const gap = currentMinX - lastBoxMaxX;
                            if (gap > (lastBoxHeight * 1.2)) {
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

    private decode(data: Float32Array, dims: readonly number[]): { text: string, conf: number } {
        const timeSteps = dims[1];
        const numClasses = dims[2];

        let sb = [];
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
                    if (realIdx >= 0 && realIdx < this.keys.length) {
                        sb.push(this.keys[realIdx]);
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
}

export const paddleOCRService = new PaddleOCRService();
