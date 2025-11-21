/**
 * Person Re-Identification Detector using Deep Learning (ONNX.js)
 * 
 * Implements State-of-the-Art (SOTA) Re-ID model based on:
 * - OSNet (Omni-Scale Network) architecture for lightweight deployment
 * - 512-dimensional feature embeddings
 * - Cosine similarity for person matching
 * 
 * Features:
 * - Real-time feature extraction from person crops
 * - Robust to illumination, pose, and viewpoint changes
 * - Memory-efficient with ONNX.js WebAssembly runtime
 */

export class ReIDDetector {
    constructor(modelPath = null) {
        this.modelPath = modelPath || '/models/osnet_ain_x1_0.onnx'; // Default OSNet model path
        this.session = null;
        this.inputShape = [1, 3, 256, 128]; // Standard Re-ID input: 256x128 (height x width)
        this.inputHeight = 256;
        this.inputWidth = 128;
        this.featureDim = 512; // Standard Re-ID feature dimension
        
        // ImageNet normalization (standard for Re-ID models)
        this.mean = [0.485, 0.456, 0.406];
        this.std = [0.229, 0.224, 0.225];
    }

    /**
     * Load Re-ID model from ONNX file
     */
    async load() {
        try {
            const ort = await import('onnxruntime-web');
            
            // Configure ONNX.js environment for Re-ID
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = false;
            
            const providers = ['wasm'];

            console.log('Loading Re-ID model from:', this.modelPath);
            console.log('Re-ID ONNX Runtime configuration:', {
                numThreads: ort.env.wasm.numThreads,
                simd: ort.env.wasm.simd,
                executionProviders: providers,
                inputShape: this.inputShape,
                featureDim: this.featureDim
            });
            
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: providers,
                graphOptimizationLevel: 'all'
            });

            if (!this.session) {
                throw new Error('Re-ID session creation returned null');
            }

            console.log('✓ Re-ID model loaded successfully');
            console.log('Re-ID Input names:', this.session.inputNames);
            console.log('Re-ID Output names:', this.session.outputNames);
            
            return true;
        } catch (error) {
            console.warn('Failed to load Re-ID model:', error);
            console.warn('Re-ID model path:', this.modelPath);
            console.warn('Falling back to histogram-based features');
            this.session = null;
            // Don't throw error - allow fallback to histogram features
            return false;
        }
    }

    /**
     * Preprocess person crop image for Re-ID model
     * - Resize to 256x128 (height x width) preserving aspect ratio
     * - Pad with mean value to maintain aspect ratio
     * - Normalize with ImageNet statistics
     * - Convert to CHW format
     */
    preprocess(imageData, box, canvas) {
        const [x1, y1, x2, y2] = box.map(v => Math.round(v));
        const width = x2 - x1;
        const height = y2 - y1;
        
        if (width <= 0 || height <= 0) {
            return null;
        }

        // Create temporary canvas for person crop
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = width;
        cropCanvas.height = height;
        const cropCtx = cropCanvas.getContext('2d');
        
        // Draw person crop
        cropCtx.drawImage(
            imageData,
            x1, y1, width, height,
            0, 0, width, height
        );

        // Resize to Re-ID input size with aspect ratio preservation
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = this.inputWidth;
        resizeCanvas.height = this.inputHeight;
        const resizeCtx = resizeCanvas.getContext('2d');
        
        // Calculate scaling to fit while preserving aspect ratio
        const scale = Math.min(
            this.inputWidth / width,
            this.inputHeight / height
        );
        const scaledWidth = Math.round(width * scale);
        const scaledHeight = Math.round(height * scale);
        
        // Center the scaled image
        const offsetX = Math.round((this.inputWidth - scaledWidth) / 2);
        const offsetY = Math.round((this.inputHeight - scaledHeight) / 2);
        
        // Fill with mean color (gray)
        resizeCtx.fillStyle = `rgb(${this.mean[0] * 255}, ${this.mean[1] * 255}, ${this.mean[2] * 255})`;
        resizeCtx.fillRect(0, 0, this.inputWidth, this.inputHeight);
        
        // Draw scaled person crop centered
        resizeCtx.drawImage(
            cropCanvas,
            0, 0, width, height,
            offsetX, offsetY, scaledWidth, scaledHeight
        );

        // Get image data
        const imageDataResized = resizeCtx.getImageData(0, 0, this.inputWidth, this.inputHeight);
        const data = imageDataResized.data;

        // Convert RGBA to RGB, normalize with ImageNet stats, and transpose to CHW
        const tensor = new Float32Array(3 * this.inputHeight * this.inputWidth);
        
        for (let i = 0; i < data.length; i += 4) {
            const pixelIdx = Math.floor(i / 4);
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Normalize: (pixel / 255.0 - mean) / std
            // Channel R
            tensor[pixelIdx] = (r / 255.0 - this.mean[0]) / this.std[0];
            // Channel G
            tensor[pixelIdx + this.inputHeight * this.inputWidth] = (g / 255.0 - this.mean[1]) / this.std[1];
            // Channel B
            tensor[pixelIdx + 2 * this.inputHeight * this.inputWidth] = (b / 255.0 - this.mean[2]) / this.std[2];
        }

        return tensor;
    }

    /**
     * Extract feature embedding from person crop
     * Returns 512-dimensional feature vector for Re-ID
     */
    async extractFeature(imageData, box, canvas) {
        // Fallback to null if model not loaded
        if (!this.session) {
            return null;
        }

        try {
            // Preprocess person crop
            const tensor = this.preprocess(imageData, box, canvas);
            if (!tensor) {
                return null;
            }

            // Create ONNX tensor
            const ort = await import('onnxruntime-web');
            const inputTensor = new ort.Tensor('float32', tensor, this.inputShape);

            // Run inference
            const feeds = {};
            feeds[this.session.inputNames[0]] = inputTensor;

            const outputs = await this.session.run(feeds);
            
            // Get feature output (usually first output)
            const outputName = this.session.outputNames[0];
            const features = outputs[outputName].data;
            
            // Normalize features (L2 normalization for cosine similarity)
            const normalizedFeatures = this.l2Normalize(features);
            
            return new Float32Array(normalizedFeatures);
        } catch (error) {
            console.error('Re-ID feature extraction error:', error);
            return null;
        }
    }

    /**
     * L2 normalize feature vector for cosine similarity
     */
    l2Normalize(features) {
        let norm = 0;
        for (let i = 0; i < features.length; i++) {
            norm += features[i] * features[i];
        }
        norm = Math.sqrt(norm);
        
        if (norm === 0) {
            return new Float32Array(features.length);
        }
        
        const normalized = new Float32Array(features.length);
        for (let i = 0; i < features.length; i++) {
            normalized[i] = features[i] / norm;
        }
        
        return normalized;
    }

    /**
     * Calculate cosine similarity between two feature vectors
     * Both vectors should be L2-normalized
     */
    cosineSimilarity(feat1, feat2) {
        if (!feat1 || !feat2) return 0;
        if (feat1.length !== feat2.length) return 0;
        
        let dotProduct = 0;
        for (let i = 0; i < feat1.length; i++) {
            dotProduct += feat1[i] * feat2[i];
        }
        
        // Since vectors are normalized, cosine similarity = dot product
        return Math.max(0, Math.min(1, dotProduct));
    }

    /**
     * Check if Re-ID model is loaded and ready
     */
    isReady() {
        return this.session !== null;
    }

    /**
     * Set custom model path (useful for loading different Re-ID models)
     */
    setModelPath(modelPath) {
        this.modelPath = modelPath;
    }
}

