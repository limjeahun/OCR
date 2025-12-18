/**
 * TensorFlow.js Document Classification Model Training Script
 * 
 * Uses Transfer Learning with MobileNet to classify:
 * - ID_CARD (주민등록증)
 * - DRIVER_LICENSE (운전면허증)
 * - BUSINESS_REGISTRATION (사업자등록증)
 * 
 * Usage: node scripts/train-classifier.js
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp'); // For image processing

// Configuration
const CONFIG = {
    IMAGE_SIZE: 224,  // MobileNet input size
    BATCH_SIZE: 16,
    EPOCHS: 20,
    LEARNING_RATE: 0.001,
    VALIDATION_SPLIT: 0.2,
    CLASSES: ['id_card', 'driver_license', 'business_reg'],
    TRAINING_DATA_DIR: path.join(__dirname, '..', 'training-data'),
    MODEL_OUTPUT_DIR: path.join(__dirname, '..', 'public', 'models', 'classifier')
};

/**
 * Load and preprocess an image
 */
async function loadImage(imagePath) {
    try {
        // Read and resize image to 224x224
        const imageBuffer = await sharp(imagePath)
            .resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, {
                fit: 'cover',
                position: 'center'
            })
            .removeAlpha()
            .raw()
            .toBuffer();

        // Convert to tensor and normalize to [-1, 1] (MobileNet preprocessing)
        const tensor = tf.tensor3d(
            new Uint8Array(imageBuffer),
            [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3]
        );

        // MobileNet expects values in [-1, 1]
        return tensor.div(127.5).sub(1);
    } catch (error) {
        console.error(`Error loading image ${imagePath}:`, error.message);
        return null;
    }
}

/**
 * Load all training data
 */
async function loadTrainingData() {
    console.log('\n📂 Loading training data...');

    const images = [];
    const labels = [];

    for (let classIndex = 0; classIndex < CONFIG.CLASSES.length; classIndex++) {
        const className = CONFIG.CLASSES[classIndex];
        const classDir = path.join(CONFIG.TRAINING_DATA_DIR, className);

        if (!fs.existsSync(classDir)) {
            console.warn(`⚠️  Directory not found: ${classDir}`);
            continue;
        }

        const files = fs.readdirSync(classDir).filter(f =>
            /\.(jpg|jpeg|png|webp)$/i.test(f)
        );

        console.log(`  📁 ${className}: ${files.length} images`);

        for (const file of files) {
            const imagePath = path.join(classDir, file);
            const tensor = await loadImage(imagePath);

            if (tensor) {
                images.push(tensor);
                labels.push(classIndex);
            }
        }
    }

    if (images.length === 0) {
        throw new Error('No images loaded! Check your training-data directory.');
    }

    console.log(`\n✅ Total images loaded: ${images.length}`);

    // Stack images into a single tensor
    const xs = tf.stack(images);

    // Clean up individual tensors
    images.forEach(t => t.dispose());

    // One-hot encode labels
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), CONFIG.CLASSES.length);

    return { xs, ys };
}

/**
 * Create the classification model using MobileNet as base
 */
async function createModel() {
    console.log('\n🔧 Creating model with MobileNet backbone...');

    // Load pre-trained MobileNet (without top classification layer)
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
    );

    console.log('  ✅ MobileNet loaded');

    // Get the output of the second-to-last layer
    // MobileNet v1 0.25 224: We'll use the global average pooling output
    const layer = mobilenet.getLayer('conv_pw_13_relu');

    // Create truncated model (feature extractor)
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });

    // Freeze the base model layers
    truncatedMobilenet.layers.forEach(layer => {
        layer.trainable = false;
    });

    // Build the full model with custom classification head
    const model = tf.sequential();

    // Add MobileNet feature extractor
    model.add(tf.layers.inputLayer({ inputShape: [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3] }));

    // We need to apply MobileNet as a function
    // Instead, let's build a proper model

    // Create classification head
    const input = tf.input({ shape: [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3] });

    // Apply feature extractor
    const features = truncatedMobilenet.apply(input);

    // Global Average Pooling
    const pooled = tf.layers.globalAveragePooling2d().apply(features);

    // Dense layers
    const dense1 = tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }).apply(pooled);

    const dropout = tf.layers.dropout({ rate: 0.5 }).apply(dense1);

    const output = tf.layers.dense({
        units: CONFIG.CLASSES.length,
        activation: 'softmax'
    }).apply(dropout);

    const model2 = tf.model({ inputs: input, outputs: output });

    // Compile the model
    model2.compile({
        optimizer: tf.train.adam(CONFIG.LEARNING_RATE),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log('  ✅ Model created and compiled');
    model2.summary();

    return model2;
}

/**
 * Train the model
 */
async function trainModel(model, xs, ys) {
    console.log('\n🚀 Starting training...');
    console.log(`  Batch size: ${CONFIG.BATCH_SIZE}`);
    console.log(`  Epochs: ${CONFIG.EPOCHS}`);
    console.log(`  Validation split: ${CONFIG.VALIDATION_SPLIT * 100}%`);

    const history = await model.fit(xs, ys, {
        batchSize: CONFIG.BATCH_SIZE,
        epochs: CONFIG.EPOCHS,
        validationSplit: CONFIG.VALIDATION_SPLIT,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `  Epoch ${epoch + 1}/${CONFIG.EPOCHS} - ` +
                    `loss: ${logs.loss.toFixed(4)}, ` +
                    `acc: ${(logs.acc * 100).toFixed(1)}%, ` +
                    `val_loss: ${logs.val_loss.toFixed(4)}, ` +
                    `val_acc: ${(logs.val_acc * 100).toFixed(1)}%`
                );
            }
        }
    });

    return history;
}

/**
 * Save the model
 */
async function saveModel(model) {
    console.log('\n💾 Saving model...');

    // Create output directory
    if (!fs.existsSync(CONFIG.MODEL_OUTPUT_DIR)) {
        fs.mkdirSync(CONFIG.MODEL_OUTPUT_DIR, { recursive: true });
    }

    // Save model in TensorFlow.js format
    const savePath = `file://${CONFIG.MODEL_OUTPUT_DIR}`;
    await model.save(savePath);

    // Save class names for reference
    const classesPath = path.join(CONFIG.MODEL_OUTPUT_DIR, 'classes.json');
    fs.writeFileSync(classesPath, JSON.stringify({
        classes: CONFIG.CLASSES,
        classLabels: {
            'id_card': 'ID_CARD',
            'driver_license': 'DRIVER_LICENSE',
            'business_reg': 'BUSINESS_REGISTRATION'
        }
    }, null, 2));

    console.log(`  ✅ Model saved to: ${CONFIG.MODEL_OUTPUT_DIR}`);
    console.log(`  📄 Files created:`);
    console.log(`     - model.json`);
    console.log(`     - group1-shard*of*.bin`);
    console.log(`     - classes.json`);
}

/**
 * Main training pipeline
 */
async function main() {
    console.log('═'.repeat(60));
    console.log('  TensorFlow.js Document Classifier Training');
    console.log('═'.repeat(60));

    try {
        // Load data
        const { xs, ys } = await loadTrainingData();

        // Create model
        const model = await createModel();

        // Train model
        await trainModel(model, xs, ys);

        // Save model
        await saveModel(model);

        // Cleanup
        xs.dispose();
        ys.dispose();

        console.log('\n' + '═'.repeat(60));
        console.log('  ✅ Training completed successfully!');
        console.log('═'.repeat(60));
        console.log('\nNext steps:');
        console.log('1. The model is saved in public/models/classifier/');
        console.log('2. Update tensorflowService.ts to load this model');
        console.log('3. Test classification with your app');

    } catch (error) {
        console.error('\n❌ Training failed:', error);
        process.exit(1);
    }
}

main();
