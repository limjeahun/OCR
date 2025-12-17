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

export interface OCRPipelineResult extends OCRResult {
    documentType: string;
    processedImage?: string; // Data URL of processed image
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
