// import * as tf from '@tensorflow/tfjs'; // Removed to prevent hang
import { ClassificationResult, DocumentType } from './types';

// Placeholder for a real model URL
const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1';

export class TensorFlowService {
    // private model: tf.GraphModel | null = null;
    private model: any = null; // Type as any for now
    private isLoading = false;

    async loadModel() {
        if (this.model || this.isLoading) return;
        this.isLoading = true;
        try {
            // In a real scenario, we would load a custom model trained on ID cards
            // this.model = await tf.loadGraphModel(MODEL_URL);
            console.log('TensorFlow model loaded (Mock - Logic Only)');

            // Simulate loading time
            await new Promise(resolve => setTimeout(resolve, 500));
            this.model = true; // Mark as loaded
        } catch (error) {
            console.error('Failed to load TF model:', error);
        } finally {
            this.isLoading = false;
        }
    }

    async classify(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<ClassificationResult> {
        // Ensure "model" is loaded (just a delay for now)
        if (!this.model) {
            await this.loadModel();
        }

        // TODO: Implement actual inference with TF model
        // For now, use Aspect Ratio Heuristic which is very effective for this distinction

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

        // Use a smaller delay to feel responsive
        await new Promise(resolve => setTimeout(resolve, 200));

        // Portrait (Height > Width) -> Likely Business Registration (A4)
        if (height > width) {
            return {
                type: 'BUSINESS_REGISTRATION',
                confidence: 0.85
            };
        }

        // Landscape -> Likely ID Card or Driver License
        // For now default to ID_CARD as they are similar shape
        return {
            type: 'ID_CARD',
            confidence: 0.90
        };
    }
}

export const tensorFlowService = new TensorFlowService();
