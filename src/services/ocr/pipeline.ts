import { tensorFlowService } from './tensorflowService';
import { openCVService } from './opencvService';
// import { tesseractService } from './tesseractService'; // Replaced by PaddleOCR
import { paddleOCRService } from './paddleOCRService';
import { OCRPipelineResult } from './types';

export class OCRPipeline {
    // Master processing function
    async processImage(imageSource: HTMLImageElement, onProgress?: (status: string) => void): Promise<OCRPipelineResult> {
        // 1. Classification
        onProgress?.('Classifying Document Type...');
        console.log('Step 1: Classifying...');
        let classification;
        try {
            classification = await tensorFlowService.classify(imageSource);
            console.log('Classification:', classification);
        } catch (e) {
            console.error('Classification step failed:', e);
            throw new Error('Document classification failed. Please try again.');
        }

        // 2. Preprocessing (OpenCV)
        onProgress?.('Preprocessing Image (OpenCV)...');
        console.log('Step 2: Preprocessing...');
        // PaddleOCR (DBNet) is robust to noise, but basic deskew/resize might help.
        // For now, we pass the raw image to PaddleOCRService which handles its own resizing standard (limit 960).
        // If we want visualization, we can still run a simple preprocess.

        // Note: paddleOCRService.recognize internally handles resizing/normalization for the model.
        // We might just want to convert to canvas for consistency if needed, but it accepts ImageElement.
        // Let's obtain a clean canvas for display purposes if we want to show "processed" image, 
        // but for recognition, we pass the source.

        // Actually, to keep 'processedImage' return value consistent, let's just use the original image 
        // or a lightly processed version.
        // Let's stick to valid OpenCV inputs.

        const processedCanvas = await openCVService.preprocess(imageSource, {
            // Disable heavy preprocessing for PaddleOCR as it handles it.
            // Converting to canvas.
            blockSize: 11, // defaults
            C: 5,
            deskew: false
        });
        const processedImageDataUrl = processedCanvas.toDataURL('image/png');

        // 3. OCR (PaddleOCR)
        onProgress?.('Recognizing Text (PaddleOCR)... This may take a moment.');
        console.log('Step 3: Recognizing text (PaddleOCR)...');
        // We pass the RAW imageSource to PaddleOCR for best quality, 
        // or processedCanvas if we really trust our preprocessing.
        // Usually deep learning models prefer raw images (maybe scaled).
        // Let's pass imageSource.
        const result = await paddleOCRService.recognize(imageSource, onProgress);

        // 4. Parsing (if applicable)
        onProgress?.(`Parsing extracted text for ${classification.type}...`);
        let extractedData;
        if (classification.type === 'BUSINESS_REGISTRATION') {
            const { businessRegistrationParser } = await import('./parsers/businessRegistrationParser');
            extractedData = businessRegistrationParser.parse(result.text);
        } else if (classification.type === 'ID_CARD') {
            const { idCardParser } = await import('./parsers/idCardParser');
            extractedData = idCardParser.parse(result.text);
        } else if (classification.type === 'DRIVER_LICENSE') {
            const { driverLicenseParser } = await import('./parsers/driverLicenseParser');
            extractedData = driverLicenseParser.parse(result.text);
        }

        return {
            documentType: classification.type,
            text: result.text,
            processedImage: processedImageDataUrl,
            confidence: result.confidence,
            extractedData
        };
    }
}

export const ocrPipeline = new OCRPipeline();
