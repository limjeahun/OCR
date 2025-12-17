import { createWorker, Worker, PSM } from 'tesseract.js';
import { OCRResult } from './types';

export class TesseractService {
    private worker: Worker | null = null;
    private isInitializing = false;

    async init() {
        if (this.worker || this.isInitializing) return;
        this.isInitializing = true;

        try {
            // Switch to 'kor' ONLY to prevent English hallucinations (e.g. (F)SRUEAA, AST)
            // 'kor' usually includes digits and basic punctuation.
            this.worker = await createWorker('kor');

            // Set parameters to improve accuracy
            await this.worker.setParameters({
                tessedit_pageseg_mode: PSM.SINGLE_BLOCK,
                preserve_interword_spaces: '1',
            });
            console.log('Tesseract worker initialized');
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
