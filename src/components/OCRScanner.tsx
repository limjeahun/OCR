"use client";

import React, { useState } from 'react';
import { ImageUploader } from './ImageUploader';
import { ResultDisplay } from './ResultDisplay';
import { ocrApi } from '@/services/api/ocrApi';
import { tensorFlowService, ImageQualityResult } from '@/services/ocr/tensorflowService';
import { OCRPipelineResult, BusinessRegistrationData, IdCardData, DriverLicenseData, DocumentType } from '@/services/ocr/types';
import { BusinessRegistrationForm } from './BusinessRegistrationForm';
import { IdCardForm } from './IdCardForm';
import { DriverLicenseForm } from './DriverLicenseForm';
import { Play, AlertTriangle, XCircle } from 'lucide-react';

export type BusinessType = 'INDIVIDUAL' | 'CORPORATE';

// 분류 결과 상태 (OCR 전 단계)
interface ClassificationState {
    type: DocumentType;
    confidence: number;
    imageBase64: string;
}

export const OCRScanner = () => {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<OCRPipelineResult | null>(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [businessType, setBusinessType] = useState<BusinessType>('CORPORATE');

    // 분류 완료 후 OCR 대기 상태
    const [classification, setClassification] = useState<ClassificationState | null>(null);
    const [awaitingBusinessType, setAwaitingBusinessType] = useState(false);

    // 에러 및 경고 상태
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [qualityWarning, setQualityWarning] = useState<ImageQualityResult | null>(null);

    // Step 1: 이미지 업로드 시 문서 타입 분류만 수행
    const handleImageSelect = async (file: File) => {
        const imageUrl = URL.createObjectURL(file);
        setSelectedImage(imageUrl);
        setResult(null);
        setClassification(null);
        setAwaitingBusinessType(false);
        setErrorMessage(null);
        setQualityWarning(null);
        setIsLoading(true);

        try {
            // Load image
            const img = new Image();
            img.src = imageUrl;
            await new Promise((resolve) => { img.onload = resolve; });

            // Step 1: 문서 타입 분류 (로컬 TensorFlow.js)
            setStatusMessage('문서 타입 분류 중...');
            const classResult = await tensorFlowService.classify(img);
            console.log('Classification:', classResult);

            // Step 1.5: UNKNOWN 타입 체크 (유효하지 않은 문서)
            if (classResult.type === 'UNKNOWN') {
                setErrorMessage('운전면허증, 주민등록증, 사업자 등록증 이미지만 가능합니다.');
                setIsLoading(false);
                setStatusMessage('');
                return;
            }

            // Step 1.6: 이미지 품질 분석
            setStatusMessage('이미지 품질 분석 중...');
            const qualityResult = tensorFlowService.analyzeImageQuality(img);
            console.log('Image Quality:', qualityResult);

            // Step 2: 이미지 리사이징 및 압축 (OCR 정확도 유지)
            setStatusMessage('이미지 최적화 중...');
            const maxDimension = 2500; // 최대 2500px (OCR 정확도 유지)
            let targetWidth = img.width;
            let targetHeight = img.height;

            // 이미지가 너무 크면 리사이징
            if (img.width > maxDimension || img.height > maxDimension) {
                const ratio = Math.min(maxDimension / img.width, maxDimension / img.height);
                targetWidth = Math.round(img.width * ratio);
                targetHeight = Math.round(img.height * ratio);
                console.log(`Image resized: ${img.width}x${img.height} → ${targetWidth}x${targetHeight}`);
            }

            const canvas = document.createElement('canvas');
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(img, 0, 0, targetWidth, targetHeight);

            // JPEG로 압축 (품질 90% - OCR 정확도 유지)
            const imageBase64 = canvas.toDataURL('image/jpeg', 0.9);
            console.log(`Image size: ${Math.round(imageBase64.length / 1024)}KB`);

            // 분류 결과 저장
            setClassification({
                type: classResult.type,
                confidence: classResult.confidence,
                imageBase64,
            });

            // 이미지 품질이 낮으면 경고 표시 (진행 여부는 사용자 선택)
            if (qualityResult.isLowQuality) {
                setQualityWarning(qualityResult);
                setIsLoading(false);
                setStatusMessage('');
                return;
            }

            // 사업자등록증이면 유형 선택 대기
            if (classResult.type === 'BUSINESS_REGISTRATION') {
                setAwaitingBusinessType(true);
                setStatusMessage('사업자 유형을 선택한 후 OCR 시작 버튼을 클릭하세요.');
                setIsLoading(false);
            } else {
                // 다른 문서 타입은 바로 OCR 진행
                await runOcr(classResult.type, imageBase64, 'CORPORATE');
            }
        } catch (error) {
            console.error('Classification Error:', error);
            setStatusMessage(`오류 발생: ${error instanceof Error ? error.message : '알 수 없는 오류'}`);
            setIsLoading(false);
        }
    };

    // 품질 경고 무시하고 진행
    const handleProceedAnyway = () => {
        setQualityWarning(null);
        if (classification) {
            if (classification.type === 'BUSINESS_REGISTRATION') {
                setAwaitingBusinessType(true);
                setStatusMessage('사업자 유형을 선택한 후 OCR 시작 버튼을 클릭하세요.');
            } else {
                runOcr(classification.type, classification.imageBase64, 'CORPORATE');
            }
        }
    };

    // Step 2: OCR 실행 (사업자등록증은 유형 선택 후 호출)
    const runOcr = async (docType: DocumentType, imageBase64: string, bizType: BusinessType) => {
        setIsLoading(true);
        setAwaitingBusinessType(false);

        try {
            // Backend API 호출 - OCR 요청
            setStatusMessage('OCR 요청 중...');
            const requestId = await ocrApi.submitOcr(imageBase64, docType, bizType);
            console.log('OCR Request ID:', requestId);

            // 결과 폴링
            const apiResult = await ocrApi.pollResult(requestId, setStatusMessage);
            console.log('OCR Result:', apiResult);

            // parsedData가 비어있으면 rawText에서 JSON 파싱 시도
            let dataToConvert = apiResult.parsedData;
            if (!dataToConvert || Object.keys(dataToConvert).length === 0) {
                // rawText가 JSON 형태인 경우 파싱
                if (apiResult.rawText) {
                    try {
                        const parsed = JSON.parse(apiResult.rawText);
                        if (typeof parsed === 'object' && parsed !== null) {
                            dataToConvert = parsed;
                            console.log('Parsed from rawText:', dataToConvert);
                        }
                    } catch {
                        console.log('rawText is not JSON, using as-is');
                    }
                }
            }

            // API 응답 → OCRPipelineResult 변환
            const extractedData = convertParsedDataToExtractedData(docType, dataToConvert || {});

            setResult({
                documentType: docType,
                text: apiResult.rawText || '',
                confidence: 0.9,
                rawText: apiResult.rawText,
                extractedData,
                requestId: apiResult.requestId,
            });
        } catch (error) {
            console.error('OCR Error:', error);
            const errorMsg = error instanceof Error ? error.message : '알 수 없는 오류';

            // LOW_QUALITY 에러 특별 처리
            if (errorMsg.startsWith('OCR_LOW_QUALITY:')) {
                const message = errorMsg.replace('OCR_LOW_QUALITY:', '');
                setErrorMessage(message);
                setStatusMessage('');
            } else {
                setStatusMessage(`오류 발생: ${errorMsg}`);
            }
        } finally {
            setIsLoading(false);
            setStatusMessage('');
        }
    };

    // 사업자 유형 선택 후 OCR 시작 버튼 클릭
    const handleStartOcr = () => {
        if (classification) {
            runOcr(classification.type, classification.imageBase64, businessType);
        }
    };

    const handleClear = () => {
        setSelectedImage(null);
        setResult(null);
        setClassification(null);
        setAwaitingBusinessType(false);
        setErrorMessage(null);
        setQualityWarning(null);
    };

    return (
        <div className="w-full max-w-4xl mx-auto space-y-8 pb-20">
            {/* ... Header ... */}

            <div className="bg-card border rounded-2xl shadow-sm p-6 md:p-8 space-y-8">
                <ImageUploader
                    onImageSelect={handleImageSelect}
                    selectedImage={selectedImage}
                    onClear={handleClear}
                    isLoading={isLoading}
                />

                {/* 에러 메시지 (유효하지 않은 문서 유형) */}
                {errorMessage && (
                    <div className="p-6 bg-red-50 rounded-xl border border-red-200 space-y-4">
                        <div className="flex items-center gap-3 text-red-600">
                            <XCircle className="w-6 h-6" />
                            <p className="text-lg font-semibold">{errorMessage}</p>
                        </div>
                        <p className="text-sm text-slate-600">
                            다른 이미지를 업로드해 주세요.
                        </p>
                    </div>
                )}

                {/* 이미지 품질 경고 */}
                {qualityWarning && (
                    <div className="p-6 bg-yellow-50 rounded-xl border border-yellow-300 space-y-4">
                        <div className="flex items-center gap-3 text-yellow-700">
                            <AlertTriangle className="w-6 h-6" />
                            <p className="text-lg font-semibold">이미지 품질이 낮습니다</p>
                        </div>
                        <p className="text-sm text-slate-700">
                            {qualityWarning.recommendation || '더 선명한 이미지로 다시 시도하면 OCR 정확도가 향상됩니다.'}
                        </p>
                        <div className="flex flex-col sm:flex-row gap-3 justify-center">
                            <button
                                type="button"
                                onClick={handleClear}
                                className="px-6 py-3 bg-white text-slate-700 rounded-lg font-medium border border-slate-300 hover:bg-slate-100 transition-colors"
                            >
                                새 이미지 업로드
                            </button>
                            <button
                                type="button"
                                onClick={handleProceedAnyway}
                                className="px-6 py-3 bg-yellow-600 text-white rounded-lg font-medium hover:bg-yellow-700 transition-colors"
                            >
                                그래도 진행
                            </button>
                        </div>
                    </div>
                )}

                {/* 사업자등록증 유형 선택 (OCR 전 단계) */}
                {awaitingBusinessType && classification?.type === 'BUSINESS_REGISTRATION' && (
                    <div className="p-6 bg-blue-50 rounded-xl border border-blue-200 space-y-4">
                        <div className="text-center">
                            <p className="text-lg font-semibold text-slate-800">
                                문서 타입: <span className="text-blue-600">사업자등록증</span>
                            </p>
                            <p className="text-sm text-slate-600 mt-1">
                                사업자 유형을 선택하고 OCR을 시작하세요.
                            </p>
                        </div>

                        <div className="flex justify-center gap-3">
                            <button
                                type="button"
                                onClick={() => setBusinessType('INDIVIDUAL')}
                                className={`px-6 py-3 rounded-lg text-sm font-medium transition-colors ${businessType === 'INDIVIDUAL'
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : 'bg-white text-slate-600 hover:bg-slate-100 border'
                                    }`}
                            >
                                개인사업자
                            </button>
                            <button
                                type="button"
                                onClick={() => setBusinessType('CORPORATE')}
                                className={`px-6 py-3 rounded-lg text-sm font-medium transition-colors ${businessType === 'CORPORATE'
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : 'bg-white text-slate-600 hover:bg-slate-100 border'
                                    }`}
                            >
                                법인사업자
                            </button>
                        </div>

                        <div className="flex justify-center">
                            <button
                                type="button"
                                onClick={handleStartOcr}
                                className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors flex items-center gap-2 shadow-lg"
                            >
                                <Play size={20} />
                                OCR 시작
                            </button>
                        </div>
                    </div>
                )}

                <ResultDisplay
                    result={result}
                    isLoading={isLoading}
                    statusMessage={statusMessage}
                />

                {/* Correction Form */}
                {result?.extractedData && !isLoading && (
                    <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
                        {result.documentType === 'BUSINESS_REGISTRATION' && (
                            <BusinessRegistrationForm
                                data={result.extractedData as BusinessRegistrationData}
                                requestId={result.requestId || ''}
                                businessType={businessType}
                            />
                        )}
                        {result.documentType === 'ID_CARD' && (
                            <IdCardForm data={result.extractedData as IdCardData} />
                        )}
                        {result.documentType === 'DRIVER_LICENSE' && (
                            <DriverLicenseForm data={result.extractedData as DriverLicenseData} />
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * Backend API parsedData → Frontend extractedData 변환
 */
function convertParsedDataToExtractedData(
    documentType: string,
    parsedData: Record<string, string>
): BusinessRegistrationData | IdCardData | DriverLicenseData | undefined {
    if (documentType === 'BUSINESS_REGISTRATION') {
        return {
            // Backend: businessNumber → Frontend: registrationNumber
            registrationNumber: parsedData['businessNumber'] || parsedData['registrationNumber'] || '',
            // Backend: merchantName → Frontend: corporateName
            corporateName: parsedData['merchantName'] || parsedData['corporateName'] || '',
            // Backend: representativeName → Frontend: representative
            representative: parsedData['representativeName'] || parsedData['representative'] || '',
            // Backend: openingDate → Frontend: establishmentDate
            establishmentDate: parsedData['openingDate'] || parsedData['establishmentDate'] || '',
            // Backend: corporateNumber → Frontend: corporateRegistrationNumber
            corporateRegistrationNumber: parsedData['corporateNumber'] || parsedData['corporateRegistrationNumber'] || '',
            // Backend: address → Frontend: businessAddress
            businessAddress: parsedData['address'] || parsedData['businessAddress'] || '',
            // Backend: headOfficeAddress → Frontend: headAddress
            headAddress: parsedData['headOfficeAddress'] || parsedData['headAddress'] || '',
        };
    } else if (documentType === 'ID_CARD') {
        return {
            name: parsedData['name'] || '',
            rrn: parsedData['rrn'] || parsedData['residentNumber'] || '',
            address: parsedData['address'] || '',
            issueDate: parsedData['issueDate'] || '',
        };
    } else if (documentType === 'DRIVER_LICENSE') {
        return {
            name: parsedData['name'] || '',
            rrn: parsedData['rrn'] || '',
            licenseNumber: parsedData['licenseNumber'] || '',
            type: parsedData['licenseType'] || parsedData['type'] || '',  // licenseType → type
            address: parsedData['address'] || '',
            issueDate: parsedData['issueDate'] || '',
            code: parsedData['serialNumber'] || parsedData['code'] || '',  // serialNumber → code
        };
    }
    return undefined;
}
