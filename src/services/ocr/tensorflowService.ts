import { ClassificationResult, DocumentType } from './types';

// TensorFlow.js import for Super Resolution
// Note: Using dynamic import to avoid blocking initial load
let tf: any = null;

// Model URL for Super Resolution (Real-ESRGAN Lite - TFLite converted)
// Alternative: Use OpenCV-based enhancement as fallback
const SR_MODEL_URL = '/models/sr_model/model.json';

export interface EnhancementResult {
    canvas: HTMLCanvasElement;
    enhanced: boolean;
    method: 'super_resolution' | 'opencv_sharpen' | 'none';
    processingTime: number;
}

// Class name mapping
const CLASS_NAMES: DocumentType[] = ['ID_CARD', 'DRIVER_LICENSE', 'BUSINESS_REGISTRATION'];
const CLASSIFIER_MODEL_URL = '/models/classifier/model.json';

export class TensorFlowService {
    private model: any = null;  // Classification model
    private srModel: any = null;
    private isLoading = false;
    private isSRLoading = false;
    private srModelAvailable = false;
    private classifierAvailable = false;

    async loadModel() {
        if (this.model || this.isLoading) return;
        this.isLoading = true;
        try {
            // Try to load the trained classifier model
            if (!tf) {
                tf = await import('@tensorflow/tfjs');
            }

            try {
                this.model = await tf.loadLayersModel(CLASSIFIER_MODEL_URL);
                this.classifierAvailable = true;
                console.log('[TensorFlow] Classification model loaded successfully');
            } catch (e) {
                console.warn('[TensorFlow] Classification model not found, using heuristic fallback');
                this.model = 'heuristic';  // Mark as fallback mode
                this.classifierAvailable = false;
            }
        } catch (error) {
            console.error('Failed to load TF model:', error);
            this.model = 'heuristic';
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Load TensorFlow.js and Super Resolution model
     */
    async loadSuperResolutionModel(): Promise<boolean> {
        if (this.srModel || this.isSRLoading) return this.srModelAvailable;
        this.isSRLoading = true;

        try {
            // Dynamic import TensorFlow.js
            if (!tf) {
                tf = await import('@tensorflow/tfjs');
                console.log('[TensorFlow.js] Loaded');
            }

            // Try to load Super Resolution model if available
            try {
                this.srModel = await tf.loadGraphModel(SR_MODEL_URL);
                this.srModelAvailable = true;
                console.log('[Super Resolution] Model loaded');
            } catch (e) {
                console.warn('[Super Resolution] Model not available, using OpenCV enhancement fallback');
                this.srModelAvailable = false;
            }

            return this.srModelAvailable;
        } catch (error) {
            console.error('[TensorFlow.js] Failed to load:', error);
            return false;
        } finally {
            this.isSRLoading = false;
        }
    }

    async classify(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<ClassificationResult> {
        if (!this.model) {
            await this.loadModel();
        }

        let width, height;
        if (imageElement instanceof HTMLImageElement) {
            width = imageElement.naturalWidth;
            height = imageElement.naturalHeight;
        } else {
            width = imageElement.width;
            height = imageElement.height;
        }

        const aspectRatio = width / height;
        console.log(`Classifying: ${width}x${height} (Aspect: ${aspectRatio.toFixed(2)})`);

        // If trained model is available, use it
        if (this.classifierAvailable && this.model && typeof this.model !== 'string') {
            try {
                console.log('[TensorFlow] Using trained classifier model');

                // Preprocess image for MobileNet
                const tensor = tf.tidy(() => {
                    const imageTensor = tf.browser.fromPixels(imageElement);
                    const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
                    // Normalize to [-1, 1] (MobileNet preprocessing)
                    const normalized = resized.div(127.5).sub(1);
                    return normalized.expandDims(0);  // Add batch dimension
                });

                // Predict
                const predictions = this.model.predict(tensor);
                const probabilities = await predictions.data();

                // Find class with highest probability
                let maxProb = 0;
                let maxIndex = 0;
                for (let i = 0; i < probabilities.length; i++) {
                    if (probabilities[i] > maxProb) {
                        maxProb = probabilities[i];
                        maxIndex = i;
                    }
                }

                // Cleanup
                tensor.dispose();
                predictions.dispose();

                const predictedType = CLASS_NAMES[maxIndex];
                console.log(`[TensorFlow] Predicted: ${predictedType} (${(maxProb * 100).toFixed(1)}%)`);

                return { type: predictedType, confidence: maxProb };
            } catch (error) {
                console.error('[TensorFlow] Classification error, falling back to heuristic:', error);
            }
        }

        // Fallback: Aspect ratio heuristic
        console.log('[TensorFlow] Using aspect ratio heuristic fallback');

        // Portrait -> Business Registration (A4)
        if (height > width) {
            return { type: 'BUSINESS_REGISTRATION', confidence: 0.85 };
        }

        // Landscape -> ID Card (default for now, will be refined by OCR text)
        return { type: 'ID_CARD', confidence: 0.75 };
    }

    /**
     * Enhance image quality using Super Resolution or OpenCV fallback
     */
    async enhanceImage(
        imageElement: HTMLImageElement | HTMLCanvasElement,
        onProgress?: (status: string) => void
    ): Promise<EnhancementResult> {
        const startTime = performance.now();

        onProgress?.('Loading enhancement engine...');

        // Try to use TensorFlow.js Super Resolution first
        const srAvailable = await this.loadSuperResolutionModel();

        if (srAvailable && this.srModel && tf) {
            onProgress?.('Applying Super Resolution (2x)...');
            try {
                const result = await this.applySuperResolution(imageElement);
                return {
                    canvas: result,
                    enhanced: true,
                    method: 'super_resolution',
                    processingTime: performance.now() - startTime
                };
            } catch (e) {
                console.warn('[Super Resolution] Failed, falling back to OpenCV:', e);
            }
        }

        // Fallback to OpenCV-based enhancement
        onProgress?.('Enhancing with OpenCV...');
        const result = await this.applyOpenCVEnhancement(imageElement);
        return {
            canvas: result,
            enhanced: true,
            method: 'opencv_sharpen',
            processingTime: performance.now() - startTime
        };
    }

    /**
     * Apply TensorFlow.js Super Resolution model
     */
    private async applySuperResolution(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<HTMLCanvasElement> {
        if (!tf || !this.srModel) {
            throw new Error('Super Resolution model not available');
        }

        // Convert image to tensor
        let tensor = tf.browser.fromPixels(imageElement);

        // Normalize to [0, 1]
        tensor = tensor.toFloat().div(255);

        // Add batch dimension
        tensor = tensor.expandDims(0);

        // Run inference
        const output = await this.srModel.predict(tensor);

        // Convert back to canvas
        const outputData = await output.squeeze().mul(255).clipByValue(0, 255).cast('int32').data();
        const [height, width] = output.shape.slice(1, 3);

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(width, height);

        // RGB to RGBA
        for (let i = 0, j = 0; i < outputData.length; i += 3, j += 4) {
            imageData.data[j] = outputData[i];
            imageData.data[j + 1] = outputData[i + 1];
            imageData.data[j + 2] = outputData[i + 2];
            imageData.data[j + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);

        // Cleanup
        tensor.dispose();
        output.dispose();

        return canvas;
    }

    /**
     * OpenCV-based image enhancement (fallback)
     * Applies: Upscale 1.5x + Sharpening + Contrast Enhancement + Denoising
     */
    private async applyOpenCVEnhancement(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<HTMLCanvasElement> {
        const cv = window.cv;
        if (!cv) {
            throw new Error('OpenCV not loaded');
        }

        let src = cv.imread(imageElement);
        let dst = new cv.Mat();
        let enhanced = new cv.Mat();

        try {
            // 1. Upscale 1.5x using bicubic interpolation
            const scale = 1.5;
            const newWidth = Math.round(src.cols * scale);
            const newHeight = Math.round(src.rows * scale);
            cv.resize(src, dst, new cv.Size(newWidth, newHeight), 0, 0, cv.INTER_CUBIC);

            // 2. Convert to LAB color space for better enhancement
            let lab = new cv.Mat();
            cv.cvtColor(dst, lab, cv.COLOR_RGBA2RGB);
            cv.cvtColor(lab, lab, cv.COLOR_RGB2Lab);

            // Split LAB channels
            let labChannels = new cv.MatVector();
            cv.split(lab, labChannels);

            // 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            let clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
            let lChannel = labChannels.get(0);
            let enhancedL = new cv.Mat();
            clahe.apply(lChannel, enhancedL);

            // Update L channel
            labChannels.set(0, enhancedL);

            // Merge LAB channels
            let mergedLab = new cv.Mat();
            cv.merge(labChannels, mergedLab);

            // Convert back to RGB
            cv.cvtColor(mergedLab, enhanced, cv.COLOR_Lab2RGB);
            cv.cvtColor(enhanced, enhanced, cv.COLOR_RGB2RGBA);

            // 4. Unsharp Masking for sharpening
            let blurred = new cv.Mat();
            cv.GaussianBlur(enhanced, blurred, new cv.Size(0, 0), 3);
            cv.addWeighted(enhanced, 1.5, blurred, -0.5, 0, enhanced);

            // 5. Light denoising
            let denoised = new cv.Mat();
            cv.fastNlMeansDenoisingColored(enhanced, denoised, 3, 3, 7, 21);

            // Output to canvas
            const canvas = document.createElement('canvas');
            cv.imshow(canvas, denoised);

            // Cleanup
            src.delete();
            dst.delete();
            enhanced.delete();
            lab.delete();
            labChannels.delete();
            lChannel.delete();
            enhancedL.delete();
            mergedLab.delete();
            blurred.delete();
            denoised.delete();

            return canvas;
        } catch (e) {
            // Cleanup on error
            src.delete();
            dst.delete();
            enhanced.delete();
            console.error('[OpenCV Enhancement] Error:', e);

            // Return simple upscaled version as final fallback
            const canvas = document.createElement('canvas');
            const scale = 1.5;
            canvas.width = Math.round(imageElement.width * scale);
            canvas.height = Math.round(imageElement.height * scale);
            const ctx = canvas.getContext('2d')!;
            ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
            return canvas;
        }
    }
}

export const tensorFlowService = new TensorFlowService();
