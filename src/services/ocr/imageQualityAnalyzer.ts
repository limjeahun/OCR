/**
 * Image Quality Analyzer
 * Analyzes image quality metrics to estimate OCR success rate
 */

declare global {
    interface Window {
        cv: any;
    }
}

export interface ImageQualityResult {
    score: number;                          // 0-100 overall quality score
    resolution: 'low' | 'medium' | 'high';  // Resolution classification
    sharpness: number;                      // 0-100 sharpness score
    contrast: number;                       // 0-100 contrast score
    brightness: number;                     // 0-100 brightness score
    estimatedOCRSuccess: number;            // 0-100 estimated OCR success rate
    recommendation: string;                 // Human-readable recommendation
    needsEnhancement: boolean;              // Whether Super Resolution is recommended
}

export class ImageQualityAnalyzer {
    /**
     * Analyze image quality for OCR
     */
    async analyze(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<ImageQualityResult> {
        const cv = window.cv;
        if (!cv) {
            throw new Error('OpenCV not loaded');
        }

        let src = cv.imread(imageElement);
        let gray = new cv.Mat();

        try {
            // Convert to grayscale
            if (src.channels() === 4) {
                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            } else if (src.channels() === 3) {
                cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
            } else {
                src.copyTo(gray);
            }

            // 1. Resolution Analysis
            const width = src.cols;
            const height = src.rows;
            const pixels = width * height;
            const resolution = this.classifyResolution(pixels);
            const resolutionScore = this.getResolutionScore(pixels);

            // 2. Sharpness Analysis (Laplacian Variance)
            const sharpness = this.analyzeSharpness(gray, cv);

            // 3. Contrast Analysis (Standard Deviation)
            const contrast = this.analyzeContrast(gray, cv);

            // 4. Brightness Analysis
            const brightness = this.analyzeBrightness(gray, cv);

            // Calculate overall score
            const score = this.calculateOverallScore({
                resolutionScore,
                sharpness,
                contrast,
                brightness
            });

            // Estimate OCR success rate
            const estimatedOCRSuccess = this.estimateOCRSuccess(score, sharpness, contrast);

            // Generate recommendation
            const { recommendation, needsEnhancement } = this.generateRecommendation(score, sharpness, contrast);

            return {
                score: Math.round(score),
                resolution,
                sharpness: Math.round(sharpness),
                contrast: Math.round(contrast),
                brightness: Math.round(brightness),
                estimatedOCRSuccess: Math.round(estimatedOCRSuccess),
                recommendation,
                needsEnhancement
            };
        } finally {
            src.delete();
            gray.delete();
        }
    }

    private classifyResolution(pixels: number): 'low' | 'medium' | 'high' {
        if (pixels < 500000) return 'low';      // < 0.5MP
        if (pixels < 2000000) return 'medium';  // 0.5-2MP
        return 'high';                          // >= 2MP
    }

    private getResolutionScore(pixels: number): number {
        // Score based on megapixels (optimal around 2-4MP for OCR)
        const mp = pixels / 1000000;
        if (mp < 0.3) return 30;
        if (mp < 0.5) return 50;
        if (mp < 1.0) return 70;
        if (mp < 2.0) return 85;
        if (mp < 4.0) return 100;
        return 95; // Very high resolution might cause performance issues
    }

    private analyzeSharpness(gray: any, cv: any): number {
        // Use Laplacian variance as sharpness metric
        let laplacian = new cv.Mat();
        cv.Laplacian(gray, laplacian, cv.CV_64F);

        // Calculate variance
        let mean = new cv.Mat();
        let stddev = new cv.Mat();
        cv.meanStdDev(laplacian, mean, stddev);

        const variance = Math.pow(stddev.data64F[0], 2);
        laplacian.delete();
        mean.delete();
        stddev.delete();

        // Normalize to 0-100 (typical variance range: 0-5000+)
        // Higher variance = sharper image
        const normalizedScore = Math.min(100, (variance / 50) * 100);
        return normalizedScore;
    }

    private analyzeContrast(gray: any, cv: any): number {
        // Use standard deviation of pixel values as contrast metric
        let mean = new cv.Mat();
        let stddev = new cv.Mat();
        cv.meanStdDev(gray, mean, stddev);

        const std = stddev.data64F[0];
        mean.delete();
        stddev.delete();

        // Normalize to 0-100 (optimal std dev around 50-80)
        const normalizedScore = Math.min(100, (std / 64) * 100);
        return normalizedScore;
    }

    private analyzeBrightness(gray: any, cv: any): number {
        // Calculate mean brightness
        let mean = new cv.Mat();
        let stddev = new cv.Mat();
        cv.meanStdDev(gray, mean, stddev);

        const avgBrightness = mean.data64F[0];
        mean.delete();
        stddev.delete();

        // Optimal brightness around 100-180, penalize extremes
        const deviation = Math.abs(avgBrightness - 140);
        const score = Math.max(0, 100 - (deviation / 1.4));
        return score;
    }

    private calculateOverallScore(metrics: {
        resolutionScore: number;
        sharpness: number;
        contrast: number;
        brightness: number;
    }): number {
        // Weighted average - sharpness is most important for OCR
        return (
            metrics.resolutionScore * 0.2 +
            metrics.sharpness * 0.4 +
            metrics.contrast * 0.25 +
            metrics.brightness * 0.15
        );
    }

    private estimateOCRSuccess(score: number, sharpness: number, contrast: number): number {
        // OCR success correlates strongly with sharpness and contrast
        const baseSuccess = score * 0.5;
        const sharpnessBonus = sharpness * 0.3;
        const contrastBonus = contrast * 0.2;

        return Math.min(100, baseSuccess + sharpnessBonus + contrastBonus);
    }

    private generateRecommendation(score: number, sharpness: number, contrast: number): {
        recommendation: string;
        needsEnhancement: boolean;
    } {
        if (score >= 70 && sharpness >= 60) {
            return {
                recommendation: '이미지 품질이 양호합니다.',
                needsEnhancement: false
            };
        }

        if (sharpness < 40) {
            return {
                recommendation: '이미지가 흐릿합니다. Super Resolution을 적용합니다.',
                needsEnhancement: true
            };
        }

        if (contrast < 40) {
            return {
                recommendation: '대비가 낮습니다. 이미지 향상을 권장합니다.',
                needsEnhancement: true
            };
        }

        if (score < 50) {
            return {
                recommendation: '전체 품질이 낮습니다. Super Resolution을 적용합니다.',
                needsEnhancement: true
            };
        }

        return {
            recommendation: '이미지 향상으로 OCR 정확도를 높일 수 있습니다.',
            needsEnhancement: score < 60
        };
    }
}

export const imageQualityAnalyzer = new ImageQualityAnalyzer();
