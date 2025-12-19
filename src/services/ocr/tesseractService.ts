import { createWorker, Worker, PSM } from 'tesseract.js';
import { OCRResult } from './types';

export class TesseractService {
    private worker: Worker | null = null;
    private isInitializing = false;

    async init() {
        if (this.worker || this.isInitializing) return;
        this.isInitializing = true;

        try {
            // Use local paths to avoid Turbopack Worker bundling issues
            // Files are located in public/tesseract/ folder
            this.worker = await createWorker('kor', 1, {
                workerPath: '/tesseract/worker.min.js',
                corePath: '/tesseract/core',
                langPath: '/tesseract/lang',
            });

            // Set parameters to improve accuracy
            await this.worker.setParameters({
                tessedit_pageseg_mode: PSM.SINGLE_BLOCK,
                preserve_interword_spaces: '1',
            });
            console.log('Tesseract worker initialized (local paths)');
        } catch (error) {
            console.error('Failed to initialize Tesseract:', error);
            throw error;
        } finally {
            this.isInitializing = false;
        }
    }

    async recognize(image: HTMLImageElement | HTMLCanvasElement): Promise<OCRResult> {
        if (!this.worker) {
            await this.init();
        }

        if (!this.worker) throw new Error('Worker not initialized');

        try {
            const { data: { text, confidence } } = await this.worker.recognize(image);
            return {
                text,
                confidence
            };
        } catch (error) {
            console.error('OCR Recognition failed:', error);
            return { text: '', confidence: 0 };
        }
    }

    async terminate() {
        if (this.worker) {
            await this.worker.terminate();
            this.worker = null;
        }
    }
}

export const tesseractService = new TesseractService();
