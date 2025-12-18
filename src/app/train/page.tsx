'use client';

import { useState, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

const CLASSES = ['id_card', 'driver_license', 'business_reg'];
const CLASS_LABELS: Record<string, string> = {
    'id_card': 'ID_CARD (주민등록증)',
    'driver_license': 'DRIVER_LICENSE (운전면허증)',
    'business_reg': 'BUSINESS_REGISTRATION (사업자등록증)'
};

const IMAGE_SIZE = 224;
const BATCH_SIZE = 8;
const EPOCHS = 15;

interface TrainingImage {
    file: File;
    dataUrl: string;
    className: string;
}

interface TrainingLog {
    epoch: number;
    loss: number;
    acc: number;
    val_loss: number;
    val_acc: number;
}

export default function TrainPage() {
    const [images, setImages] = useState<TrainingImage[]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('');
    const [logs, setLogs] = useState<TrainingLog[]>([]);
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Handle file selection for each class
    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>, className: string) => {
        const files = Array.from(e.target.files || []);

        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                setImages(prev => [...prev, {
                    file,
                    dataUrl: ev.target?.result as string,
                    className
                }]);
            };
            reader.readAsDataURL(file);
        });
    }, []);

    // Load image as tensor
    const loadImageAsTensor = async (dataUrl: string): Promise<tf.Tensor3D | null> => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const tensor = tf.tidy(() => {
                    const imageTensor = tf.browser.fromPixels(img);
                    const resized = tf.image.resizeBilinear(imageTensor, [IMAGE_SIZE, IMAGE_SIZE]);
                    // Normalize to [-1, 1] for MobileNet
                    return resized.div(127.5).sub(1) as tf.Tensor3D;
                });
                resolve(tensor);
            };
            img.onerror = () => resolve(null);
            img.src = dataUrl;
        });
    };

    // Create model with MobileNet backbone
    const createModel = async (): Promise<tf.LayersModel> => {
        setStatus('Loading MobileNet backbone...');

        // Load MobileNet
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
        );

        // Get feature extraction layer
        const layer = mobilenet.getLayer('conv_pw_13_relu');
        const truncatedMobilenet = tf.model({
            inputs: mobilenet.inputs,
            outputs: layer.output
        });

        // Freeze base layers
        truncatedMobilenet.layers.forEach(l => { l.trainable = false; });

        // Build classification head
        const input = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3] });
        const features = truncatedMobilenet.apply(input) as tf.SymbolicTensor;
        const pooled = tf.layers.globalAveragePooling2d({}).apply(features) as tf.SymbolicTensor;
        const dense1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(pooled) as tf.SymbolicTensor;
        const dropout = tf.layers.dropout({ rate: 0.5 }).apply(dense1) as tf.SymbolicTensor;
        const output = tf.layers.dense({ units: CLASSES.length, activation: 'softmax' }).apply(dropout) as tf.SymbolicTensor;

        const model = tf.model({ inputs: input, outputs: output });

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    };

    // Start training
    const startTraining = async () => {
        if (images.length === 0) {
            alert('Please add training images first!');
            return;
        }

        // Check class balance
        const classCounts = CLASSES.map(c => images.filter(i => i.className === c).length);
        if (classCounts.some(c => c === 0)) {
            alert('Please add images for all 3 classes!');
            return;
        }

        setIsTraining(true);
        setLogs([]);
        setProgress(0);

        try {
            // Load all images as tensors
            setStatus(`Loading ${images.length} images...`);
            const tensors: tf.Tensor3D[] = [];
            const labels: number[] = [];

            for (let i = 0; i < images.length; i++) {
                const tensor = await loadImageAsTensor(images[i].dataUrl);
                if (tensor) {
                    tensors.push(tensor);
                    labels.push(CLASSES.indexOf(images[i].className));
                }
                setProgress((i / images.length) * 20);
            }

            setStatus('Preparing training data...');
            const xs = tf.stack(tensors);
            tensors.forEach(t => t.dispose());

            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), CLASSES.length);

            // Create model
            setStatus('Creating model...');
            const newModel = await createModel();
            setProgress(30);

            // Train
            setStatus('Training...');
            await newModel.fit(xs, ys, {
                batchSize: BATCH_SIZE,
                epochs: EPOCHS,
                validationSplit: 0.2,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const log: TrainingLog = {
                            epoch: epoch + 1,
                            loss: logs?.loss || 0,
                            acc: logs?.acc || 0,
                            val_loss: logs?.val_loss || 0,
                            val_acc: logs?.val_acc || 0
                        };
                        setLogs(prev => [...prev, log]);
                        setProgress(30 + ((epoch + 1) / EPOCHS) * 60);
                        setStatus(`Epoch ${epoch + 1}/${EPOCHS} - acc: ${(log.acc * 100).toFixed(1)}%`);
                    }
                }
            });

            // Cleanup
            xs.dispose();
            ys.dispose();

            setModel(newModel);
            setProgress(100);
            setStatus('Training complete! You can now download the model.');

        } catch (error) {
            console.error('Training error:', error);
            setStatus(`Error: ${error}`);
        } finally {
            setIsTraining(false);
        }
    };


    // Download trained model with proper file names
    const downloadModel = async () => {
        if (!model) return;

        setStatus('Saving model...');

        try {
            // Save model to get the artifacts
            const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
                // Create model.json download
                const modelJSON = {
                    modelTopology: artifacts.modelTopology,
                    weightsManifest: [{
                        paths: ['model.weights.bin'],
                        weights: artifacts.weightSpecs
                    }],
                    format: 'layers-model',
                    generatedBy: 'Document Classifier Training',
                    convertedBy: null
                };

                const jsonBlob = new Blob([JSON.stringify(modelJSON)], { type: 'application/json' });
                const jsonUrl = URL.createObjectURL(jsonBlob);
                const jsonLink = document.createElement('a');
                jsonLink.href = jsonUrl;
                jsonLink.download = 'model.json';
                jsonLink.click();
                URL.revokeObjectURL(jsonUrl);

                // Create weights.bin download
                if (artifacts.weightData) {
                    // Handle both ArrayBuffer and ArrayBuffer[] types
                    const weightData = artifacts.weightData;
                    const weightsBlob = new Blob(
                        Array.isArray(weightData) ? weightData : [weightData],
                        { type: 'application/octet-stream' }
                    );
                    const weightsUrl = URL.createObjectURL(weightsBlob);
                    const weightsLink = document.createElement('a');
                    weightsLink.href = weightsUrl;
                    weightsLink.download = 'model.weights.bin';

                    // Small delay to ensure sequential downloads
                    await new Promise(resolve => setTimeout(resolve, 500));
                    weightsLink.click();
                    URL.revokeObjectURL(weightsUrl);
                }

                return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
            }));

            setStatus('✅ 다운로드 완료! model.json과 model.weights.bin 파일을 public/models/classifier/ 폴더로 이동하세요.');
        } catch (error) {
            console.error('Model save error:', error);
            setStatus(`Error: ${error}`);
        }
    };

    const getImageCountByClass = (className: string) =>
        images.filter(i => i.className === className).length;

    return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <div className="max-w-6xl mx-auto">
                <h1 className="text-3xl font-bold mb-2">🧠 Document Classifier Training</h1>
                <p className="text-gray-400 mb-8">
                    Train a TensorFlow.js model to classify ID cards, driver licenses, and business registrations.
                </p>

                {/* Image Upload Section */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    {CLASSES.map(className => (
                        <div key={className} className="bg-gray-800 rounded-xl p-6">
                            <h3 className="font-semibold mb-2">{CLASS_LABELS[className]}</h3>
                            <p className="text-2xl font-bold text-blue-400 mb-4">
                                {getImageCountByClass(className)} images
                            </p>
                            <label className="block">
                                <span className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg cursor-pointer inline-block">
                                    + Add Images
                                </span>
                                <input
                                    type="file"
                                    multiple
                                    accept="image/*"
                                    className="hidden"
                                    onChange={(e) => handleFileSelect(e, className)}
                                    disabled={isTraining}
                                />
                            </label>
                        </div>
                    ))}
                </div>

                {/* Training Controls */}
                <div className="bg-gray-800 rounded-xl p-6 mb-8">
                    <div className="flex items-center gap-4 mb-4">
                        <button
                            onClick={startTraining}
                            disabled={isTraining || images.length === 0}
                            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold"
                        >
                            {isTraining ? '⏳ Training...' : '🚀 Start Training'}
                        </button>

                        {model && (
                            <button
                                onClick={downloadModel}
                                className="bg-purple-600 hover:bg-purple-700 px-6 py-3 rounded-lg font-semibold"
                            >
                                💾 Download Model
                            </button>
                        )}

                        <button
                            onClick={() => setImages([])}
                            disabled={isTraining}
                            className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-3 rounded-lg"
                        >
                            🗑️ Clear All
                        </button>
                    </div>

                    {/* Progress */}
                    {(isTraining || progress > 0) && (
                        <div className="mb-4">
                            <div className="flex justify-between text-sm text-gray-400 mb-1">
                                <span>{status}</span>
                                <span>{progress.toFixed(0)}%</span>
                            </div>
                            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-300"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Training Logs */}
                    {logs.length > 0 && (
                        <div className="mt-4">
                            <h4 className="font-semibold mb-2">Training Logs</h4>
                            <div className="bg-gray-900 rounded-lg p-4 max-h-60 overflow-y-auto font-mono text-sm">
                                {logs.map((log, i) => (
                                    <div key={i} className="text-gray-300">
                                        Epoch {log.epoch}/{EPOCHS} -
                                        loss: {log.loss.toFixed(4)},
                                        acc: {(log.acc * 100).toFixed(1)}%,
                                        val_loss: {log.val_loss.toFixed(4)},
                                        val_acc: {(log.val_acc * 100).toFixed(1)}%
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Image Preview */}
                {images.length > 0 && (
                    <div className="bg-gray-800 rounded-xl p-6">
                        <h3 className="font-semibold mb-4">Loaded Images ({images.length})</h3>
                        <div className="grid grid-cols-6 md:grid-cols-10 gap-2 max-h-60 overflow-y-auto">
                            {images.slice(0, 50).map((img, i) => (
                                <div key={i} className="relative group">
                                    <img
                                        src={img.dataUrl}
                                        alt={`${img.className} ${i}`}
                                        className="w-16 h-16 object-cover rounded-lg"
                                    />
                                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-[10px] text-center py-0.5 rounded-b-lg">
                                        {img.className.split('_')[0]}
                                    </div>
                                </div>
                            ))}
                            {images.length > 50 && (
                                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center text-sm">
                                    +{images.length - 50}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Instructions */}
                <div className="mt-8 bg-gray-800 rounded-xl p-6">
                    <h3 className="font-semibold mb-4">📋 Instructions</h3>
                    <ol className="list-decimal list-inside space-y-2 text-gray-300">
                        <li>Add training images for each document type (minimum 20 per class recommended)</li>
                        <li>Click <strong>Start Training</strong> to train the model</li>
                        <li>Wait for training to complete (watch progress and logs)</li>
                        <li>Click <strong>Download Model</strong> to save the trained model</li>
                        <li>Move the downloaded files to <code className="bg-gray-700 px-2 py-0.5 rounded">public/models/classifier/</code></li>
                        <li>Restart the app to use the new classifier</li>
                    </ol>
                </div>
            </div>
        </div>
    );
}
