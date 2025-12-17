/* eslint-disable @next/next/no-img-element */
import React from 'react';
import { OCRPipelineResult } from '@/services/ocr/types';
import { CheckCircle2, FileText, Fingerprint, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResultDisplayProps {
    result: OCRPipelineResult | null;
    isLoading: boolean;
    statusMessage: string;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, isLoading, statusMessage }) => {
    if (isLoading) {
        return (
            <div className="w-full flex flex-col items-center justify-center p-12 space-y-4 animate-in fade-in slide-in-from-bottom-4 bg-muted/20 rounded-xl border border-border/50">
                <Loader2 className="w-10 h-10 text-primary animate-spin" />
                <p className="text-muted-foreground animate-pulse text-sm font-medium">{statusMessage}</p>
            </div>
        );
    }

    if (!result) return null;

    return (
        <div className="w-full space-y-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Classification Result */}
                <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-4">
                    <div className="flex items-center space-x-3 text-primary">
                        <Fingerprint className="w-6 h-6" />
                        <h3 className="font-semibold">Document Type</h3>
                    </div>
                    <div className="text-2xl font-bold tracking-tight">
                        {result.documentType.replace('_', ' ')}
                    </div>
                    <Badge>Confidence: {Math.round(result.confidence)}%</Badge>
                </div>

                {/* Processed Image Preview (Debug) */}
                <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-4">
                    <div className="flex items-center space-x-3 text-primary">
                        <FileText className="w-6 h-6" />
                        <h3 className="font-semibold">Processed View</h3>
                    </div>
                    <div className="aspect-video rounded-lg overflow-hidden bg-black/5 border">
                        <img src={result.processedImage} alt="Processed" className="w-full h-full object-contain filter grayscale contrast-125" />
                    </div>
                </div>
            </div>

            {/* Structured Data Extraction */}
            {result.extractedData && (
                <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-4">
                    <div className="flex items-center space-x-3 text-primary">
                        <CheckCircle2 className="w-6 h-6" />
                        <h3 className="font-semibold">Extracted Information</h3>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(result.extractedData).map(([key, value]) => (
                            <div key={key} className="space-y-1">
                                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                    {key.replace(/([A-Z])/g, ' $1').trim()}
                                </p>
                                <p className="font-mono text-sm font-medium bg-muted/30 p-2 rounded border">
                                    {value || '-'}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Extracted Text */}
            <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3 text-primary">
                        <FileText className="w-6 h-6" />
                        <h3 className="font-semibold">Raw Extracted Text</h3>
                    </div>
                </div>

                <div className="relative">
                    <textarea
                        readOnly
                        className="w-full min-h-[200px] p-4 rounded-lg bg-muted/50 border focus:border-primary transition-colors font-mono text-sm resize-y"
                        value={result.text}
                    />
                </div>
            </div>
        </div>
    );
};

// Simple internal Badge component
const Badge = ({ children }: { children: React.ReactNode }) => (
    <span className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 border-transparent bg-primary text-primary-foreground hover:bg-primary/80">
        {children}
    </span>
);
