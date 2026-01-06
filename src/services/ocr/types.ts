export interface OCRResult {
    text: string;
    confidence: number;
    extractedData?: BusinessRegistrationData | IdCardData | DriverLicenseData;
}

export type DocumentType = 'BUSINESS_REGISTRATION' | 'ID_CARD' | 'DRIVER_LICENSE' | 'UNKNOWN';

export interface ClassificationResult {
    type: DocumentType;
    confidence: number;
}

export interface IdCardData {
    name: string;
    rrn: string; // Resident Registration Number
    address: string;
    issueDate: string;
}

export interface DriverLicenseData {
    name: string;
    rrn: string;
    licenseNumber: string;
    type: string;
    address: string;
    issueDate: string;
    code: string; // The code at the bottom right usually
}

export interface ImageQualityInfo {
    score: number;
    resolution: 'low' | 'medium' | 'high';
    sharpness: number;
    contrast: number;
    estimatedOCRSuccess: number;
    recommendation: string;
    enhanced: boolean;
    enhancementMethod?: 'super_resolution' | 'opencv_sharpen' | 'none';
    enhancementTime?: number;
}

export interface TextCorrectionInfo {
    applied: boolean;
    count: number;
    confidence: number;
}

export interface OCRPipelineResult extends OCRResult {
    documentType: string;
    processedImage?: string; // Data URL of processed image
    imageQuality?: ImageQualityInfo; // Quality analysis results
    rawText?: string; // Original OCR text before correction
    textCorrection?: TextCorrectionInfo; // Text correction info
    requestId?: string; // Backend OCR request ID (for save)
}

/**
 * Backend API OCR 결과 타입
 * OCRScanner에서 Backend API 응답을 변환하여 사용
 */
export interface OcrApiResult {
    requestId: string;
    documentType: DocumentType;
    rawText?: string;
    parsedData: Record<string, string>;
    extractedData?: BusinessRegistrationData | IdCardData | DriverLicenseData;
}

export interface BusinessRegistrationData {
    registrationNumber: string;
    corporateName: string;
    representative: string;
    establishmentDate: string;
    corporateRegistrationNumber: string;
    // Updated to split addresses
    businessAddress: string;   // 사업장소재지
    headAddress: string;       // 본점소재지
    // Legacy support if needed, or remove 'address' entirely if we successfully migrated parser.
    // Parser now returns businessAddress/headAddress.
    address?: string;
}
