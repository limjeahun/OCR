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

                {/* Image Quality Analysis (NEW) */}
                {result.imageQuality && (
                    <div className="p-6 rounded-xl border bg-card text-card-foreground shadow-sm space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3 text-primary">
                                <CheckCircle2 className="w-6 h-6" />
                                <h3 className="font-semibold">Image Quality</h3>
                            </div>
                            {result.imageQuality.enhanced && (
                                <Badge variant="success">Enhanced</Badge>
                            )}
                        </div>

                        {/* Quality Score */}
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span>Overall Score</span>
                                <span className="font-semibold">{result.imageQuality.score}%</span>
                            </div>
                            <ProgressBar value={result.imageQuality.score} />
                        </div>

                        {/* OCR Success Estimate */}
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span>Estimated OCR Success</span>
                                <span className="font-semibold">{result.imageQuality.estimatedOCRSuccess}%</span>
                            </div>
                            <ProgressBar value={result.imageQuality.estimatedOCRSuccess} variant={result.imageQuality.estimatedOCRSuccess > 60 ? 'success' : 'warning'} />
                        </div>

                        {/* Details */}
                        <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="text-center p-2 bg-muted/30 rounded">
                                <div className="text-muted-foreground">Sharpness</div>
                                <div className="font-semibold">{result.imageQuality.sharpness}%</div>
                            </div>
                            <div className="text-center p-2 bg-muted/30 rounded">
                                <div className="text-muted-foreground">Contrast</div>
                                <div className="font-semibold">{result.imageQuality.contrast}%</div>
                            </div>
                            <div className="text-center p-2 bg-muted/30 rounded">
                                <div className="text-muted-foreground">Resolution</div>
                                <div className="font-semibold capitalize">{result.imageQuality.resolution}</div>
                            </div>
                        </div>

                        {/* Enhancement Info */}
                        {result.imageQuality.enhanced && (
                            <div className="text-xs text-muted-foreground bg-muted/30 p-2 rounded">
                                ✨ Enhanced with {result.imageQuality.enhancementMethod === 'super_resolution' ? 'Super Resolution' : 'OpenCV'}
                                ({result.imageQuality.enhancementTime}ms)
                            </div>
                        )}

                        {/* Recommendation */}
                        <p className="text-xs text-muted-foreground">
                            {result.imageQuality.recommendation}
                        </p>
                    </div>
                )}

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

// Simple internal Badge component with variant support
const Badge = ({ children, variant = 'default' }: { children: React.ReactNode; variant?: 'default' | 'success' | 'warning' }) => {
    const variantClasses = {
        default: 'bg-primary text-primary-foreground',
        success: 'bg-green-500 text-white',
        warning: 'bg-yellow-500 text-white'
    };

    return (
        <span className={cn(
            "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 border-transparent hover:opacity-80",
            variantClasses[variant]
        )}>
            {children}
        </span>
    );
};

// Simple internal ProgressBar component
const ProgressBar = ({ value, variant = 'default' }: { value: number; variant?: 'default' | 'success' | 'warning' }) => {
    const variantClasses = {
        default: 'bg-primary',
        success: 'bg-green-500',
        warning: 'bg-yellow-500'
    };

    const bgColor = value >= 70 ? variantClasses.success : value >= 40 ? variantClasses.default : variantClasses.warning;

    return (
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
                className={cn("h-full transition-all duration-500", variant !== 'default' ? variantClasses[variant] : bgColor)}
                style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
            />
        </div>
    );
};
