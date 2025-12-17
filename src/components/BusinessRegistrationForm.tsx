"use client";

import React, { useEffect } from 'react';
import { BusinessRegistrationData } from '@/services/ocr/parsers/businessRegistrationParser';
import { Save, Copy } from 'lucide-react';

interface Props {
    data: BusinessRegistrationData;
}

export const BusinessRegistrationForm: React.FC<Props> = ({ data }) => {
    const [formData, setFormData] = React.useState<BusinessRegistrationData>(data);

    // Update form when data prop changes (new scan)
    useEffect(() => {
        setFormData(data);
    }, [data]);

    const handleChange = (key: keyof BusinessRegistrationData, value: string) => {
        setFormData(prev => ({ ...prev, [key]: value }));
    };

    const handleCopy = () => {
        const text = Object.entries(formData)
            .map(([k, v]) => `${k}: ${v}`)
            .join('\n');
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    };

    const handleSave = () => {
        console.log('Saved data:', formData);
        alert('Data saved! (Simulation)');
    };

    return (
        <div className="w-full bg-slate-50 border rounded-xl overflow-hidden shadow-sm">
            <div className="bg-slate-100 p-4 border-b flex justify-between items-center">
                <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                    사업자등록증 정보 수정
                </h3>
                <div className="flex gap-2">
                    <button
                        onClick={handleCopy}
                        className="p-2 text-slate-600 hover:text-blue-600 transition-colors"
                        title="Copy text"
                    >
                        <Copy size={18} />
                    </button>
                    <button
                        onClick={handleSave}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
                    >
                        <Save size={16} />
                        저장 완료
                    </button>
                </div>
            </div>

            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <InputField
                    label="등록번호"
                    value={formData.registrationNumber}
                    onChange={v => handleChange('registrationNumber', v)}
                />
                <InputField
                    label="법인등록번호"
                    value={formData.corporateRegistrationNumber}
                    onChange={v => handleChange('corporateRegistrationNumber', v)}
                />
                <InputField
                    label="법인명 (단체명)"
                    value={formData.corporateName}
                    onChange={v => handleChange('corporateName', v)}
                    fullWidth
                />
                <InputField
                    label="대표자"
                    value={formData.representative}
                    onChange={v => handleChange('representative', v)}
                />
                <InputField
                    label="개업연월일"
                    value={formData.establishmentDate}
                    onChange={v => handleChange('establishmentDate', v)}
                />
                <InputField
                    label="사업장 소재지"
                    value={formData.businessAddress}
                    onChange={v => handleChange('businessAddress', v)}
                    fullWidth
                    multiline
                />
                <InputField
                    label="본점 소재지"
                    value={formData.headAddress}
                    onChange={v => handleChange('headAddress', v)}
                    fullWidth
                    multiline
                />
            </div>
        </div>
    );
};

interface InputFieldProps {
    label: string;
    value: string;
    onChange: (val: string) => void;
    fullWidth?: boolean;
    multiline?: boolean;
}

const InputField: React.FC<InputFieldProps> = ({ label, value, onChange, fullWidth, multiline }) => (
    <div className={`space-y-1.5 ${fullWidth ? 'col-span-1 md:col-span-2' : ''}`}>
        <label className="text-sm font-medium text-slate-600">{label}</label>
        {multiline ? (
            <textarea
                className="flex w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 min-h-[80px] resize-y"
                value={value}
                onChange={e => onChange(e.target.value)}
            />
        ) : (
            <input
                className="flex h-10 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                value={value}
                onChange={e => onChange(e.target.value)}
            />
        )}
    </div>
);
