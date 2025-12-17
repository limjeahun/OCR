"use client";

import React, { useState } from 'react';
import { ImageUploader } from './ImageUploader';
import { ResultDisplay } from './ResultDisplay';
import { ocrPipeline } from '@/services/ocr/pipeline';
import { OCRPipelineResult } from '@/services/ocr/types';
import { ScanLine } from 'lucide-react';
import { BusinessRegistrationForm } from './BusinessRegistrationForm';
import { IdCardForm } from './IdCardForm';
import { DriverLicenseForm } from './DriverLicenseForm';
import { BusinessRegistrationData, IdCardData, DriverLicenseData } from '@/services/ocr/types';

export const OCRScanner = () => {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<OCRPipelineResult | null>(null);
    const [statusMessage, setStatusMessage] = useState('');

    const handleImageSelect = async (file: File) => {
        // Create Preview URL
        const imageUrl = URL.createObjectURL(file);
        setSelectedImage(imageUrl);
        setResult(null);
        setIsLoading(true);

        try {
            // Load image object for processing
            const img = new Image();
            img.src = imageUrl;
            await new Promise((resolve) => { img.onload = resolve; });

            // Pipeline Steps
            setStatusMessage('Initializing TensorFlow.js...');
            await new Promise(r => setTimeout(r, 500)); // UI pacing

            setStatusMessage('Classifying Document Type...');
            // Note: The pipeline calls are inside here, but we can also update status progressively 
            // if we refactor pipeline to emit events. For now, we wrap the whole call.

            const pipelineResult = await ocrPipeline.processImage(img, setStatusMessage);
            setResult(pipelineResult);

        } catch (error) {
            console.error('OCR Error:', error);
            setStatusMessage('Error occurred during processing.');
        } finally {
            setIsLoading(false);
            setStatusMessage('');
        }
    };

    const handleClear = () => {
        setSelectedImage(null);
        setResult(null);
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

                <ResultDisplay
                    result={result}
                    isLoading={isLoading}
                    statusMessage={statusMessage}
                />

                {/* Correction Form */}
                {result?.extractedData && !isLoading && (
                    <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
                        {result.documentType === 'BUSINESS_REGISTRATION' && (
                            <BusinessRegistrationForm data={result.extractedData as BusinessRegistrationData} />
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
