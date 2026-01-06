/**
 * OCR API Service
 * Backend API 연동 모듈
 */

// API 기본 URL (환경변수 또는 기본값)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

/**
 * Backend API 응답 타입
 */
export interface OcrApiResponse {
    requestId: string;
    status: 'PROCESSING' | 'COMPLETED' | 'FAILED';
    rawText?: string;
    parsedData: Record<string, string>;
}

/**
 * 가맹점 저장 요청 데이터
 */
export interface MerchantSaveData {
    requestId: string;
    businessType: 'INDIVIDUAL' | 'CORPORATE';
    merchantName: string;
    businessNumber: string;
    representativeName: string;
    address: string;
    businessCategory?: string;
    businessItem?: string;
    openingDate?: string;
}

/**
 * OCR API 서비스 클래스
 */
class OcrApiService {
    private baseUrl: string;

    constructor() {
        this.baseUrl = API_BASE_URL;
    }

    /**
     * OCR 요청 제출
     * @param imageBase64 이미지 데이터 (Base64 Data URI)
     * @param documentType 문서 유형 (BUSINESS_LICENSE, ID_CARD)
     * @param businessType 사업자 유형 (INDIVIDUAL, CORPORATE)
     * @returns 요청 ID
     */
    async submitOcr(
        imageBase64: string,
        documentType: string,
        businessType: string = 'CORPORATE'
    ): Promise<string> {
        // 문서 타입 매핑 (Frontend → Backend)
        const mappedDocType = documentType === 'BUSINESS_REGISTRATION'
            ? 'BUSINESS_LICENSE'
            : documentType;

        const response = await fetch(`${this.baseUrl}/api/v1/orc/request`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                imageUrl: imageBase64,
                type: mappedDocType,
                businessType: businessType,
            }),
        });

        if (!response.ok) {
            throw new Error(`OCR 요청 실패: ${response.status} ${response.statusText}`);
        }

        const requestId = await response.text();
        return requestId;
    }

    /**
     * OCR 결과 조회
     * @param requestId 요청 ID
     * @returns OCR 결과
     */
    async getResult(requestId: string): Promise<OcrApiResponse> {
        const response = await fetch(`${this.baseUrl}/api/v1/orc/result/${requestId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`결과 조회 실패: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * OCR 결과 폴링 (완료될 때까지)
     * @param requestId 요청 ID
     * @param onProgress 진행 상태 콜백
     * @param maxRetries 최대 재시도 횟수
     * @param intervalMs 폴링 간격 (밀리초)
     * @returns 완료된 OCR 결과
     */
    async pollResult(
        requestId: string,
        onProgress?: (status: string) => void,
        maxRetries: number = 100,  // 100회 재시도
        intervalMs: number = 5000  // 5초 간격 → 총 약 8분 대기
    ): Promise<OcrApiResponse> {
        let retries = 0;

        while (retries < maxRetries) {
            onProgress?.(`OCR 처리 중... (${retries + 1}/${maxRetries})`);

            const result = await this.getResult(requestId);

            if (result.status === 'COMPLETED') {
                return result;
            }

            if (result.status === 'FAILED') {
                throw new Error('OCR 처리 실패');
            }

            // PROCESSING 상태면 대기 후 재시도
            await new Promise(resolve => setTimeout(resolve, intervalMs));
            retries++;
        }

        throw new Error('OCR 처리 시간 초과');
    }

    /**
     * 가맹점 정보 저장
     * @param data 저장할 가맹점 데이터
     */
    async saveMerchant(data: MerchantSaveData): Promise<void> {
        const response = await fetch(`${this.baseUrl}/api/v1/orc/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`저장 실패: ${response.status} ${response.statusText}`);
        }
    }
}

// 싱글톤 인스턴스 export
export const ocrApi = new OcrApiService();
