import { tensorFlowService } from './tensorflowService';
import { openCVService } from './opencvService';
import { paddleOCRService } from './paddleOCRService';
import { imageQualityAnalyzer } from './imageQualityAnalyzer';
import { textCorrector } from './textCorrector';
import { OCRPipelineResult, ImageQualityInfo } from './types';

export class OCRPipeline {
    // Quality threshold below which enhancement is applied
    // Lowered from 60 to 40 to avoid over-processing decent quality images
    private readonly ENHANCEMENT_THRESHOLD = 40;

    async processImage(imageSource: HTMLImageElement, onProgress?: (status: string) => void): Promise<OCRPipelineResult> {
        // 0. Pre-load OpenCV (required for quality analysis)
        onProgress?.('Loading OpenCV...');
        try {
            await openCVService.loadOpenCV();
        } catch (e) {
            console.warn('OpenCV pre-load failed:', e);
        }

        // 1. Image Quality Analysis (NEW)
        onProgress?.('Analyzing Image Quality...');
        console.log('Step 1: Analyzing image quality...');

        let qualityInfo: ImageQualityInfo;
        let processableImage: HTMLImageElement | HTMLCanvasElement = imageSource;

        try {
            const qualityResult = await imageQualityAnalyzer.analyze(imageSource);
            console.log('Quality Analysis:', qualityResult);

            qualityInfo = {
                score: qualityResult.score,
                resolution: qualityResult.resolution,
                sharpness: qualityResult.sharpness,
                contrast: qualityResult.contrast,
                estimatedOCRSuccess: qualityResult.estimatedOCRSuccess,
                recommendation: qualityResult.recommendation,
                enhanced: false
            };

            // 2. Image Enhancement - Only for truly low quality images
            // Changed: Only enhance if BOTH needsEnhancement is true AND score is below threshold
            // This prevents over-processing of decent quality images
            const shouldEnhance = qualityResult.needsEnhancement && qualityResult.score < this.ENHANCEMENT_THRESHOLD;

            if (shouldEnhance) {
                onProgress?.(`Image Quality: ${qualityResult.score}% - Enhancing...`);
                console.log('Step 2: Applying image enhancement...');

                const enhancementResult = await tensorFlowService.enhanceImage(imageSource, onProgress);
                processableImage = enhancementResult.canvas;

                qualityInfo.enhanced = true;
                qualityInfo.enhancementMethod = enhancementResult.method;
                qualityInfo.enhancementTime = Math.round(enhancementResult.processingTime);

                console.log(`Enhancement applied: ${enhancementResult.method} (${qualityInfo.enhancementTime}ms)`);
            } else {
                onProgress?.(`Image Quality: ${qualityResult.score}% - Good`);
            }
        } catch (e) {
            console.warn('Quality analysis failed, proceeding without enhancement:', e);
            qualityInfo = {
                score: 50,
                resolution: 'medium',
                sharpness: 50,
                contrast: 50,
                estimatedOCRSuccess: 50,
                recommendation: 'Quality analysis unavailable',
                enhanced: false
            };
        }

        // 3. Classification
        onProgress?.('Classifying Document Type...');
        console.log('Step 3: Classifying...');
        let classification;
        try {
            classification = await tensorFlowService.classify(processableImage);
            console.log('Classification:', classification);
        } catch (e) {
            console.error('Classification step failed:', e);
            throw new Error('Document classification failed. Please try again.');
        }

        // 4. Preprocessing (OpenCV) - for display purposes
        onProgress?.('Preprocessing Image...');
        console.log('Step 4: Preprocessing...');

        const processedCanvas = await openCVService.preprocess(
            processableImage instanceof HTMLImageElement ? imageSource : processableImage,
            {
                blockSize: 11,
                C: 5,
                deskew: false
            }
        );
        const processedImageDataUrl = processedCanvas.toDataURL('image/png');

        // 5. OCR (PaddleOCR) with document type for dynamic threshold
        onProgress?.('Recognizing Text (PaddleOCR)...');
        console.log('Step 5: Recognizing text (PaddleOCR)...');
        const result = await paddleOCRService.recognize(processableImage, onProgress, classification.type);

        // 6. Text Correction (NEW: Local NLP-based)
        onProgress?.('Correcting OCR Text...');
        console.log('Step 6: Correcting OCR text...');
        const correctionResult = textCorrector.correct(result.text);
        const correctedText = correctionResult.corrected;

        if (correctionResult.corrections.length > 0) {
            console.log(`[TextCorrector] Applied ${correctionResult.corrections.length} corrections:`);
            correctionResult.corrections.forEach(c => {
                console.log(`  "${c.original}" → "${c.corrected}" (${c.method}, conf: ${c.confidence.toFixed(2)})`);
            });
        }

        // 7. Parsing (using corrected text)
        onProgress?.(`Parsing extracted text for ${classification.type}...`);
        let extractedData;
        if (classification.type === 'BUSINESS_REGISTRATION') {
            const { businessRegistrationParser } = await import('./parsers/businessRegistrationParser');
            extractedData = businessRegistrationParser.parse(correctedText);
        } else if (classification.type === 'ID_CARD') {
            const { idCardParser } = await import('./parsers/idCardParser');
            extractedData = idCardParser.parse(correctedText);
        } else if (classification.type === 'DRIVER_LICENSE') {
            const { driverLicenseParser } = await import('./parsers/driverLicenseParser');
            extractedData = driverLicenseParser.parse(correctedText);
        }

        return {
            documentType: classification.type,
            text: correctedText,  // Use corrected text
            rawText: result.text,  // Keep original for comparison
            processedImage: processedImageDataUrl,
            confidence: result.confidence,
            extractedData,
            imageQuality: qualityInfo,
            textCorrection: {  // NEW: Include correction info
                applied: correctionResult.corrections.length > 0,
                count: correctionResult.corrections.length,
                confidence: correctionResult.confidence
            }
        };
    }
}

export const ocrPipeline = new OCRPipeline();
