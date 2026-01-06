/* eslint-disable @next/next/no-img-element */
import React from 'react';
import { OCRPipelineResult } from '@/services/ocr/types';
import { CheckCircle2, FileText, Fingerprint, Loader2, Building2, User, Calendar, MapPin, Hash } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResultDisplayProps {
    result: OCRPipelineResult | null;
    isLoading: boolean;
    statusMessage: string;
}

// 필드명 한글 매핑
const fieldLabels: Record<string, { label: string; icon: React.ReactNode }> = {
    registrationNumber: { label: '사업자등록번호', icon: <Hash className="w-4 h-4" /> },
    corporateName: { label: '상호(법인명)', icon: <Building2 className="w-4 h-4" /> },
    representative: { label: '대표자', icon: <User className="w-4 h-4" /> },
    establishmentDate: { label: '개업연월일', icon: <Calendar className="w-4 h-4" /> },
    corporateRegistrationNumber: { label: '법인등록번호', icon: <Hash className="w-4 h-4" /> },
    businessAddress: { label: '사업장 소재지', icon: <MapPin className="w-4 h-4" /> },
    headAddress: { label: '본점 소재지', icon: <MapPin className="w-4 h-4" /> },
    name: { label: '성명', icon: <User className="w-4 h-4" /> },
    rrn: { label: '주민등록번호', icon: <Hash className="w-4 h-4" /> },
    address: { label: '주소', icon: <MapPin className="w-4 h-4" /> },
    issueDate: { label: '발급일', icon: <Calendar className="w-4 h-4" /> },
    licenseNumber: { label: '면허번호', icon: <Hash className="w-4 h-4" /> },
    type: { label: '면허종류', icon: <FileText className="w-4 h-4" /> },
    code: { label: '코드', icon: <Hash className="w-4 h-4" /> },
};

// 문서 타입 한글 매핑
const documentTypeLabels: Record<string, string> = {
    'BUSINESS_REGISTRATION': '사업자등록증',
    'ID_CARD': '주민등록증',
    'DRIVER_LICENSE': '운전면허증',
    'UNKNOWN': '알 수 없음',
};

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, isLoading, statusMessage }) => {
    if (isLoading) {
        return (
            <div className="w-full flex flex-col items-center justify-center p-12 space-y-4 animate-in fade-in slide-in-from-bottom-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-100">
                <div className="relative">
                    <div className="absolute inset-0 bg-blue-400 rounded-full blur-xl opacity-20 animate-pulse" />
                    <Loader2 className="w-12 h-12 text-blue-600 animate-spin relative" />
                </div>
                <p className="text-slate-600 animate-pulse text-sm font-medium">{statusMessage}</p>
            </div>
        );
    }

    if (!result) return null;

    return (
        <div className="w-full space-y-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
            {/* 문서 타입 카드 */}
            <div className="p-6 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-lg shadow-blue-200">
                <div className="flex items-center justify-between">
                    <div className="space-y-2">
                        <div className="flex items-center gap-2 text-blue-100">
                            <Fingerprint className="w-5 h-5" />
                            <span className="text-sm font-medium">문서 타입</span>
                        </div>
                        <h2 className="text-2xl font-bold tracking-tight">
                            {documentTypeLabels[result.documentType] || result.documentType}
                        </h2>
                    </div>
                    <div className="bg-white/20 backdrop-blur-sm px-4 py-2 rounded-full">
                        <span className="text-sm font-semibold">
                            신뢰도 {Math.round(result.confidence * 100)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* 추출된 정보 카드 */}
            {result.extractedData && (
                <div className="p-6 rounded-2xl border bg-white shadow-sm space-y-5">
                    <div className="flex items-center gap-3 pb-4 border-b">
                        <div className="p-2 bg-green-100 rounded-lg">
                            <CheckCircle2 className="w-5 h-5 text-green-600" />
                        </div>
                        <h3 className="text-lg font-semibold text-slate-800">추출된 정보</h3>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(result.extractedData).map(([key, value]) => {
                            const fieldInfo = fieldLabels[key] || { label: key, icon: <FileText className="w-4 h-4" /> };
                            const displayValue = value || '-';
                            const isAddress = key.toLowerCase().includes('address');

                            return (
                                <div
                                    key={key}
                                    className={cn(
                                        "group p-4 rounded-xl bg-slate-50 hover:bg-slate-100 transition-colors border border-slate-100",
                                        isAddress && "md:col-span-2"
                                    )}
                                >
                                    <div className="flex items-center gap-2 text-slate-500 mb-2">
                                        {fieldInfo.icon}
                                        <span className="text-xs font-medium uppercase tracking-wider">
                                            {fieldInfo.label}
                                        </span>
                                    </div>
                                    <p className={cn(
                                        "font-medium text-slate-800",
                                        isAddress ? "text-sm" : "text-base"
                                    )}>
                                        {displayValue}
                                    </p>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Raw Text (접을 수 있게) */}
            <details className="group">
                <summary className="flex items-center gap-3 p-4 rounded-xl border bg-white cursor-pointer hover:bg-slate-50 transition-colors">
                    <div className="p-2 bg-slate-100 rounded-lg group-open:bg-blue-100 transition-colors">
                        <FileText className="w-5 h-5 text-slate-600 group-open:text-blue-600" />
                    </div>
                    <span className="font-medium text-slate-700">원본 텍스트 보기</span>
                    <span className="ml-auto text-slate-400 text-sm">클릭하여 펼치기</span>
                </summary>
                <div className="mt-2 p-4 rounded-xl border bg-slate-50">
                    <pre className="text-sm text-slate-600 whitespace-pre-wrap font-mono overflow-auto max-h-[300px]">
                        {result.text}
                    </pre>
                </div>
            </details>
        </div>
    );
};
