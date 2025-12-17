/* eslint-disable @next/next/no-img-element */
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, ImageIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ImageUploaderProps {
    onImageSelect: (file: File) => void;
    selectedImage: string | null;
    onClear: () => void;
    isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
    onImageSelect,
    selectedImage,
    onClear,
    isLoading
}) => {
    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles?.[0]) {
            onImageSelect(acceptedFiles[0]);
        }
    }, [onImageSelect]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.webp']
        },
        maxFiles: 1,
        disabled: isLoading || !!selectedImage
    });

    if (selectedImage) {
        return (
            <div className="relative w-full aspect-video rounded-xl overflow-hidden border-2 border-border bg-muted/30 shadow-sm animate-in fade-in zoom-in-95 duration-300">
                <img
                    src={selectedImage}
                    alt="Selected"
                    className="w-full h-full object-contain"
                />
                {!isLoading && (
                    <button
                        onClick={onClear}
                        className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors backdrop-blur-sm"
                    >
                        <X className="w-5 h-5" />
                    </button>
                )}
            </div>
        );
    }

    return (
        <div
            {...getRootProps()}
            className={cn(
                "w-full aspect-video rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ease-in-out group",
                isDragActive
                    ? "border-primary bg-primary/5 scale-[0.99]"
                    : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/30",
                isLoading && "opacity-50 cursor-not-allowed"
            )}
        >
            <input {...getInputProps()} />
            <div className="p-4 rounded-full bg-muted group-hover:bg-background group-hover:shadow-md transition-all duration-300 mb-4">
                {isDragActive ? (
                    <Upload className="w-8 h-8 text-primary animate-bounce" />
                ) : (
                    <ImageIcon className="w-8 h-8 text-muted-foreground group-hover:text-primary transition-colors" />
                )}
            </div>
            <div className="text-center space-y-2">
                <h3 className="text-lg font-semibold text-foreground/80 group-hover:text-primary transition-colors">
                    {isDragActive ? "Drop the ID card here" : "Upload ID Card"}
                </h3>
                <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                    Drag and drop or click to upload. Supports JPG, PNG.
                </p>
            </div>
        </div>
    );
};
