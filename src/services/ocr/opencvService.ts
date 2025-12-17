declare global {
    interface Window {
        cv: any;
    }
}

export class OpenCVService {
    private isLoaded = false;

    async loadOpenCV(): Promise<void> {
        if (this.isLoaded && window.cv) return;

        return new Promise((resolve, reject) => {
            if (window.cv) {
                this.isLoaded = true;
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = '/opencv.js';
            script.async = true;
            script.onload = () => {
                // OpenCV.js sometimes requires a small delay or onRuntimeInitialized callback
                if (window.cv.getBuildInformation) {
                    this.isLoaded = true;
                    console.log('OpenCV.js loaded successfully');
                    resolve();
                } else {
                    window.cv.onRuntimeInitialized = () => {
                        this.isLoaded = true;
                        console.log('OpenCV.js runtime initialized');
                        resolve();
                    };
                }
            };
            script.onerror = () => {
                reject(new Error('Failed to load OpenCV.js'));
            };
            document.body.appendChild(script);
        });
    }

    private deskew(src: any): any {
        const cv = window.cv;
        let gray = new cv.Mat();

        // Handle input channels
        if (src.channels() === 3 || src.channels() === 4) {
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        } else {
            src.copyTo(gray);
        }

        // Invert -> White text on black background
        cv.bitwise_not(gray, gray);

        // Threshold
        let thresh = new cv.Mat();
        cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

        // Find contours instead of findNonZero (which is missing in some opencv.js builds)
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(thresh, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

        let angles: number[] = [];
        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);
            // Ignore small noise
            if (cv.contourArea(cnt) < 50) {
                cnt.delete();
                continue;
            }

            // Get rotated rect for each contour (word/blob)
            let box = cv.minAreaRect(cnt);
            let angle = box.angle;

            // Normalize angle
            // OpenCV MinAreaRect returns angle in [-90, 0) usually
            // We expect small skew around 0.
            if (angle < -45) {
                angle = 90 + angle;
            }

            angles.push(angle);
            cnt.delete();
        }

        // Calculate median or average angle
        let finalAngle = 0;
        if (angles.length > 0) {
            angles.sort((a, b) => a - b);
            const mid = Math.floor(angles.length / 2);
            finalAngle = angles[mid];
        }

        console.log(`Deskew angle found (median of ${angles.length} contours): ${finalAngle}`);

        // Rotate
        let center = new cv.Point(src.cols / 2, src.rows / 2); // Rotate around center of image
        let M = cv.getRotationMatrix2D(center, finalAngle, 1.0);
        let rotated = new cv.Mat();
        cv.warpAffine(src, rotated, M, src.size(), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar(255, 255, 255));

        // Cleanup
        gray.delete();
        thresh.delete();
        contours.delete();
        hierarchy.delete();
        M.delete();

        return rotated;
    }

    async preprocess(
        imageElement: HTMLImageElement | HTMLCanvasElement,
        config: { blockSize?: number; C?: number; deskew?: boolean } = {}
    ): Promise<HTMLCanvasElement> {
        if (!this.isLoaded) await this.loadOpenCV();

        const cv = window.cv;
        let src = cv.imread(imageElement);
        let dst = new cv.Mat();
        let gray = new cv.Mat();

        try {
            // 0. Upscale (2x) to improve OCR on small text
            // Higher helps separate strokes in '법', '명' etc.
            let dsize = new cv.Size(src.cols * 2, src.rows * 2);
            cv.resize(src, src, dsize, 0, 0, cv.INTER_CUBIC);

            // 1. Convert to Grayscale
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

            // 2. Deskew (if enabled)
            if (config.deskew) {
                // Deskew typically works best on the grayscale image
                // We return a new Mat, so we must handle memory
                let rotated = this.deskew(gray);
                gray.delete(); // delete old gray
                gray = rotated; // update reference
            }

            // 3. Simple Grayscale (Best for Tesseract)
            // User Feedback: Adaptive thresholding degraded accuracy on business registration docs.
            // Tesseract handles simple grayscale well.
            // We keep the upscaling (Step 0) and Denoising (Step 2) but skip manual binarization.

            // 3. Adaptive Thresholding (Binarization)
            // This is key for removing watermarks and shadows
            // ADAPTIVE_THRESH_GAUSSIAN_C is usually smoother than MEAN_C
            // const blockSize = config.blockSize || 31; // Must be odd
            // const C = config.C || 8; // Lowered from 15 to 8 to be safer

            // cv.adaptiveThreshold(
            //     gray,
            //     dst,
            //     255,
            //     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            //     cv.THRESH_BINARY,
            //     blockSize,
            //     C
            // );

            // 4. Slight Denoise (Median Blur) - removes salt-and-pepper noise
            // cv.medianBlur(dst, dst, 3);

            // 5. Morphological ops (Optional) - can help if text is broken
            // let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 1));
            // cv.morphologyEx(dst, dst, cv.MORPH_CLOSE, kernel);
            // kernel.delete();

            // Reverting to simple output:
            const outputCanvas = document.createElement('canvas');
            cv.imshow(outputCanvas, gray); // 'dst' is currently GaussianBlurred Grayscale

            return outputCanvas;
        } catch (e) {
            console.error("OpenCV processing failed", e);
            throw e;
        } finally {
            src.delete();
            dst.delete();
            gray.delete();
        }
    }

    // New helper for PaddleOCR DBNet Post-processing
    // mapData is Float32Array from ONNX (logits or prob), shape [H, W]
    getBoxesFromMap(mapData: Float32Array, width: number, height: number, boxThreshold: number = 0.3): any[] {
        const cv = window.cv;

        // 1. Create Mat from data
        let mat = cv.matFromArray(height, width, cv.CV_32FC1, mapData);
        let binary = new cv.Mat();

        // 2. Threshold -> Binary Map
        // DBNet output is probability map (0~1)
        cv.threshold(mat, binary, boxThreshold, 1.0, cv.THRESH_BINARY);

        // Convert to 8UC1 for findContours
        binary.convertTo(binary, cv.CV_8UC1, 255);

        // 2.5 Dilation to merge characters roughly
        // Use a horizontal kernel to merge wide-spaced chars (like 代表)
        // Reduced from 12x1 to 6x1 to prevent merging separate columns (Name + Date)
        let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(6, 1));
        cv.dilate(binary, binary, kernel);
        kernel.delete();

        // 3. Find Contours
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(binary, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

        let boxes: any[] = [];

        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);

            // Filter small blobs
            if (cv.contourArea(cnt) < 100) { // arbitrary small threshold
                cnt.delete();
                continue;
            }

            // MinAreaRect
            let rect = cv.minAreaRect(cnt);

            // DBNet Unclip (Expand box)
            // Naive approach: Scale width and height by 1.5 or 2.0?
            // Standard DBNet unclip ratio is calculated by area/perimeter.
            // Here we use a fixed scale factor which is often sufficient for simple boxes.
            // 1.5 is a safe starting point.
            // DBNet Unclip (Expand box)
            // Naive approach: Scale width and height by different factors.
            // Width: 1.5x (Safe horizontal merge)
            // Height: 1.4x (Restore vertical integrity to fix 'cutting' issues, still < 1.6 to avoid merge)
            rect.size.width *= 1.5;
            rect.size.height *= 1.4;

            // Get corners
            let vertices = cv.RotatedRect.points(rect);

            // Check confidence (mean validation)? skip for speed

            boxes.push({
                points: vertices, // [{x,y}, {x,y}, {x,y}, {x,y}]
                center: rect.center,
                size: rect.size,
                angle: rect.angle
            });

            cnt.delete();
        }

        // Cleanup
        mat.delete();
        binary.delete();
        contours.delete();
        hierarchy.delete();

        return boxes;
    }

    // Helper to sort points: TL, TR, BR, BL
    private orderPoints(points: any[]): any[] {
        // Sort by Y to separate Top/Bottom
        points.sort((a, b) => a.y - b.y);

        // Sep Top (first 2) and Bottom (last 2)
        const top = points.slice(0, 2);
        const bottom = points.slice(2, 4);

        // Sort Top by X -> TL, TR
        top.sort((a, b) => a.x - b.x);

        // Sort Bottom by X -> BL, BR (Wait, order is TL, TR, BR, BL for warp?)
        // Standard warpPerspective with 3 points usually OK, but 4 points need specific order.
        // My target in warpBox is: (0,0), (w,0), (w,h), (0,h).
        // This corresponds to: TL, TR, BR, BL.
        bottom.sort((a, b) => a.x - b.x);
        // Bottom left is index 0 of bottom? No, sorted by X. output: [BL, BR].
        // We want BR then BL? Or BL then BR?
        // Target:
        // 0: 0,0 (TL)
        // 1: w,0 (TR)
        // 2: w,h (BR)
        // 3: 0,h (BL)

        // top[0] is TL, top[1] is TR.
        // bottom[0] is BL, bottom[1] is BR.

        return [top[0], top[1], bottom[1], bottom[0]];
    }

    // Warp the text box to a straight rectangle
    // boxPoints: [{x,y}, {x,y}, {x,y}, {x,y}] (Expected: TL, TR, BR, BL after sorting)
    warpBox(src: any, boxPoints: any[], width: number, height: number): any {
        const cv = window.cv;

        // Ensure points are ordered
        const ordered = this.orderPoints(boxPoints);

        // Source points
        let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            ordered[0].x, ordered[0].y,
            ordered[1].x, ordered[1].y,
            ordered[2].x, ordered[2].y,
            ordered[3].x, ordered[3].y
        ]);

        // Destination points (0,0) -> (w, h)
        let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0, 0,
            width, 0,
            width, height,
            0, height
        ]);

        let M = cv.getPerspectiveTransform(srcTri, dstTri);
        let dst = new cv.Mat();

        cv.warpPerspective(src, dst, M, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());

        // Cleanup
        srcTri.delete();
        dstTri.delete();
        M.delete();

        return dst;
    }

    // Normalize image for model input
    // src: Mat (RGB)
    // mean: [0.485, 0.456, 0.406] (scaled or unscaled?) 
    // PaddleOCR usually requires: (pixel/255.0 - mean) / std.
    // Mean/Std default for Paddle: 0.5, 0.5 (for map 0-1) or different?
    // PP-OCRv4 Default: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], scale 1/255.
    blobFromImage(src: any, scale: number = 1 / 255.0, mean: number[] = [0.485, 0.456, 0.406], std: number[] = [0.229, 0.224, 0.225]): Float32Array {
        const cv = window.cv;

        // Convert to float
        let floatMat = new cv.Mat();
        src.convertTo(floatMat, cv.CV_32FC3, scale); // 0-255 -> 0-1

        // Split channels
        let channels = new cv.MatVector();
        cv.split(floatMat, channels);

        // Normalize per channel
        // (x - mean) / std
        for (let i = 0; i < 3; i++) {
            let ch = channels.get(i);
            // subtract mean
            let m = new cv.Mat(ch.rows, ch.cols, ch.type(), new cv.Scalar(mean[i]));
            cv.subtract(ch, m, ch);
            m.delete();

            // divide std
            let s = new cv.Mat(ch.rows, ch.cols, ch.type(), new cv.Scalar(std[i]));
            cv.divide(ch, s, ch);
            s.delete();
        }

        // Create Float32Array in CHW format
        const rows = src.rows;
        const cols = src.cols;
        const data = new Float32Array(3 * rows * cols);

        for (let c = 0; c < 3; c++) {
            let ch = channels.get(c);
            // data() returns Uint8Array usually, but for CV_32F it matches?
            // Need to copy manually or use .data32F
            let chData = ch.data32F; // Float32Array view
            data.set(chData, c * rows * cols);
            ch.delete();
        }

        floatMat.delete();
        channels.delete();

        return data; // CHW
    }
}

export const openCVService = new OpenCVService();
