/**
 * Main Application: ONNX.js Real-time Webcam Person Tracker
 * Features:
 * - Real-time person detection using ONNX.js YOLOv8
 * - Re-identification person tracking
 * - 60-second memory persistence (maintains IDs when person disappears)
 */

import { YOLOv8Detector } from './detector.js';
import { PersonTracker } from './tracker.js';

// Application state
const app = {
    video: null,
    canvas: null,
    ctx: null,
    detector: null,
    tracker: null,
    isRunning: false,
    animationId: null,
    
    // UI elements
    startBtn: null,
    stopBtn: null,
    resetBtn: null,
    statusDiv: null,
    confidenceSlider: null,
    confidenceValue: null,
    
    // Stats
    fps: 0,
    frameCount: 0,
    lastFpsTime: 0,
    totalDetections: 0,
    
    // Processing state
    isProcessing: false,
    
    // Last tracking results for continuous rendering (updated by detection)
    lastTracks: [],
    lastUpdateTime: 0,
    
    // Config
    modelPath: '/models/yolov8n.onnx', // Update this path to your model
    confidenceThreshold: 0.5,
    maxMemorySeconds: 60
};

/**
 * Initialize application
 */
async function init() {
    console.log('Initializing MOTIP Detection Web Client...');
    
    // Get UI elements
    app.video = document.getElementById('videoElement');
    app.canvas = document.getElementById('videoCanvas');
    app.ctx = app.canvas.getContext('2d');
    app.startBtn = document.getElementById('startBtn');
    app.stopBtn = document.getElementById('stopBtn');
    app.resetBtn = document.getElementById('resetBtn');
    app.statusDiv = document.getElementById('status');
    app.confidenceSlider = document.getElementById('confidenceSlider');
    app.confidenceValue = document.getElementById('confidenceValue');
    
    // Setup event listeners
    app.startBtn.addEventListener('click', startWebcam);
    app.stopBtn.addEventListener('click', stopWebcam);
    app.resetBtn.addEventListener('click', resetTracker);
    app.confidenceSlider.addEventListener('input', updateConfidenceThreshold);
    
    // Try to load model (optional - will show error if not found)
    try {
        await loadModel();
        updateStatus('ready', '✅ Model ready - Click "Start Webcam" to begin');
    } catch (error) {
        console.warn('Model not loaded yet:', error.message);
        updateStatus('idle', '⚠️ Model not found. Please place yolov8n.onnx in /public/models/');
    }
    
    // Initialize tracker
    app.tracker = new PersonTracker(app.maxMemorySeconds);
    
    console.log('✓ Application initialized');
}

/**
 * Load ONNX model
 */
async function loadModel() {
    console.log('Loading YOLOv8 model from:', app.modelPath);
    
    // Create new detector instance
    app.detector = new YOLOv8Detector(
        app.modelPath,
        app.confidenceThreshold,
        0.4 // IoU threshold
    );
    
    try {
        await app.detector.load();
        console.log('✓ Model loaded successfully');
        console.log('Model session:', app.detector.session ? 'ready' : 'not ready');
        return true;
    } catch (error) {
        console.error('Failed to load model:', error);
        app.detector = null;
        throw error;
    }
}

/**
 * Start webcam
 */
async function startWebcam() {
    try {
        updateStatus('running', '🎥 Starting webcam...');
        
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });
        
        app.video.srcObject = stream;
        
        // Wait for video to be ready
        app.video.onloadedmetadata = async () => {
            // Set canvas size to match video
            app.canvas.width = app.video.videoWidth;
            app.canvas.height = app.video.videoHeight;
            
            // Load model first if not loaded
            if (!app.detector || !app.detector.session) {
                updateStatus('running', '🔄 Loading model...');
                try {
                    await loadModel();
                    updateStatus('running', '✅ Webcam active - Tracking persons...');
                } catch (error) {
                    console.error('Failed to load model:', error);
                    updateStatus('error', '❌ Failed to load model: ' + error.message);
                    stopWebcam();
                    return;
                }
            }
            
            // Start processing only after model is loaded
            app.isRunning = true;
            app.startBtn.disabled = true;
            app.stopBtn.disabled = false;
            app.resetBtn.disabled = false;
            
            processFrame();
        };
        
        app.video.play();
        
    } catch (error) {
        console.error('Failed to start webcam:', error);
        updateStatus('error', '❌ Failed to access webcam: ' + error.message);
    }
}

/**
 * Stop webcam
 */
function stopWebcam() {
    app.isRunning = false;
    
    if (app.animationId) {
        cancelAnimationFrame(app.animationId);
        app.animationId = null;
    }
    
    if (app.video.srcObject) {
        const tracks = app.video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        app.video.srcObject = null;
    }
    
    // Clear canvas
    app.ctx.clearRect(0, 0, app.canvas.width, app.canvas.height);
    
    // Reset tracking state
    app.lastTracks = [];
    app.lastUpdateTime = 0;
    app.isProcessing = false;
    
    app.startBtn.disabled = false;
    app.stopBtn.disabled = true;
    app.resetBtn.disabled = true;
    
    updateStatus('idle', '⏸️ Webcam stopped');
}

/**
 * Reset tracker
 */
function resetTracker() {
    if (app.tracker) {
        app.tracker.reset();
        app.lastTracks = [];
        app.lastUpdateTime = 0;
        updateStatus('running', '🔄 Tracker reset - All IDs cleared');
        updateStats();
    }
}

/**
 * Update confidence threshold
 */
function updateConfidenceThreshold(e) {
    const value = parseFloat(e.target.value);
    app.confidenceThreshold = value;
    app.confidenceValue.textContent = value.toFixed(1);
    
    if (app.detector) {
        app.detector.setConfidenceThreshold(value);
    }
}

/**
 * Main processing loop - called every frame
 */
function processFrame() {
    if (!app.isRunning) return;
    
    const currentTime = performance.now() / 1000; // Convert to seconds
    
    // Always draw video frame first (every frame)
    app.ctx.drawImage(app.video, 0, 0, app.canvas.width, app.canvas.height);
    
    // Draw bounding boxes from last known tracks (every frame for smooth rendering)
    if (app.lastTracks.length > 0) {
        drawBoundingBoxes(app.lastTracks, currentTime);
    }
    
    // Trigger detection (async - runs in background)
    // This ensures detection doesn't block rendering
    if (app.detector && 
        app.detector.session && 
        app.video.readyState === app.video.HAVE_ENOUGH_DATA &&
        !app.isProcessing) {
        // Throttle detection to ~10 FPS for performance
        const timeSinceLastDetection = currentTime - app.lastUpdateTime;
        if (timeSinceLastDetection > 0.1) { // 10 FPS detection
            processDetection();
        }
    }
    
    // Calculate FPS
    app.frameCount++;
    const currentTimeMs = performance.now();
    if (currentTimeMs - app.lastFpsTime >= 1000) {
        app.fps = app.frameCount;
        app.frameCount = 0;
        app.lastFpsTime = currentTimeMs;
        updateStats();
    }
    
    // Schedule next frame (runs at ~60 FPS for smooth video)
    app.animationId = requestAnimationFrame(processFrame);
}

/**
 * Process detection and tracking
 */
async function processDetection() {
    // Prevent multiple simultaneous detections
    if (app.isProcessing) return;
    
    // Check if detector is loaded and ready
    if (!app.detector) {
        console.warn('Detector not initialized');
        return;
    }
    
    if (!app.detector.session) {
        console.warn('Detector model not loaded yet');
        return;
    }
    
    if (!app.tracker) {
        console.warn('Tracker not initialized');
        return;
    }
    
    app.isProcessing = true;
    
    try {
        const currentTime = performance.now() / 1000; // Convert to seconds
        
        // Run detection
        const result = await app.detector.detect(
            app.video,
            app.canvas.width,
            app.canvas.height
        );
        
        app.totalDetections = result.detections.length;
        
        // Update tracker
        const tracks = app.tracker.update(
            result.detections,
            app.video,
            app.canvas,
            currentTime
        );
        
        // Store tracks for continuous rendering
        app.lastTracks = tracks;
        app.lastUpdateTime = currentTime;
        
    } catch (error) {
        console.error('Detection error:', error);
        // Update status on error
        if (error.message && error.message.includes('Model not loaded')) {
            updateStatus('error', '❌ Model not loaded. Please check console.');
            // Try to reload model
            try {
                await loadModel();
                updateStatus('running', '✅ Model reloaded - Tracking resumed');
            } catch (reloadError) {
                console.error('Failed to reload model:', reloadError);
            }
        }
    } finally {
        app.isProcessing = false;
    }
}

/**
 * Draw bounding boxes and IDs - called every frame for smooth rendering
 */
function drawBoundingBoxes(tracks, currentTime) {
    let activeCount = 0;
    let memoryCount = 0;
    
    for (const track of tracks) {
        const [x1, y1, x2, y2] = track.box.map(v => Math.round(v));
        const [r, g, b] = track.color;
        
        // Validate bounding box coordinates
        if (x2 <= x1 || y2 <= y1 || x1 < 0 || y1 < 0 || 
            x2 > app.canvas.width || y2 > app.canvas.height) {
            continue; // Skip invalid boxes
        }
        
        // Determine if track is in memory (lost but within 60 seconds)
        const isInMemory = track.state === 'lost' && 
                          (currentTime - track.lastSeen) <= app.maxMemorySeconds;
        
        if (track.state === 'active') {
            activeCount++;
        } else if (isInMemory) {
            memoryCount++;
        }
        
        // Draw semi-transparent fill for better visibility
        if (isInMemory) {
            app.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.15)`;
            app.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        } else {
            app.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.1)`;
            app.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }
        
        // Draw bounding box border
        app.ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        app.ctx.lineWidth = 3;
        app.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw corner markers for better visibility
        const cornerSize = 8;
        app.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        // Top-left
        app.ctx.fillRect(x1 - 2, y1 - 2, cornerSize, 3);
        app.ctx.fillRect(x1 - 2, y1 - 2, 3, cornerSize);
        // Top-right
        app.ctx.fillRect(x2 - cornerSize + 2, y1 - 2, cornerSize, 3);
        app.ctx.fillRect(x2 - 1, y1 - 2, 3, cornerSize);
        // Bottom-left
        app.ctx.fillRect(x1 - 2, y2 - 1, cornerSize, 3);
        app.ctx.fillRect(x1 - 2, y2 - cornerSize + 2, 3, cornerSize);
        // Bottom-right
        app.ctx.fillRect(x2 - cornerSize + 2, y2 - 1, cornerSize, 3);
        app.ctx.fillRect(x2 - 1, y2 - cornerSize + 2, 3, cornerSize);
        
        // Calculate duration
        const duration = currentTime - track.created;
        const memoryTime = track.state === 'lost' ? 
                          Math.max(0, app.maxMemorySeconds - (currentTime - track.lastSeen)) : 0;
        
        // Prepare label
        let label = `ID: ${track.id}`;
        if (track.state === 'active') {
            label += ` | ${duration.toFixed(1)}s`;
        } else if (isInMemory) {
            label += ` | Memory: ${memoryTime.toFixed(1)}s`;
        }
        if (track.score) {
            label += ` | ${(track.score * 100).toFixed(0)}%`;
        }
        
        // Draw label background
        app.ctx.font = 'bold 14px Arial';
        const labelMetrics = app.ctx.measureText(label);
        const labelWidth = labelMetrics.width + 12;
        const labelHeight = 22;
        
        // Label background with rounded corners effect
        app.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
        app.ctx.fillRect(x1, y1 - labelHeight - 2, labelWidth, labelHeight);
        
        // Draw label text
        app.ctx.fillStyle = '#ffffff';
        app.ctx.font = 'bold 14px Arial';
        app.ctx.fillText(label, x1 + 6, y1 - 6);
    }
    
    // Draw statistics overlay
    drawStatsOverlay(activeCount, memoryCount, app.totalDetections);
}

/**
 * Draw statistics overlay
 */
function drawStatsOverlay(activeTracks, memoryIds, totalDetections) {
    const stats = [
        `Active: ${activeTracks}`,
        `Memory: ${memoryIds}`,
        `Detections: ${totalDetections}`,
        `FPS: ${app.fps}`
    ];
    
    app.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    app.ctx.fillRect(10, 10, 200, stats.length * 25 + 10);
    
    app.ctx.fillStyle = '#ffffff';
    app.ctx.font = 'bold 14px Arial';
    
    stats.forEach((stat, i) => {
        app.ctx.fillText(stat, 15, 30 + i * 25);
    });
}

/**
 * Update status message
 */
function updateStatus(type, message) {
    app.statusDiv.className = `status ${type}`;
    app.statusDiv.textContent = message;
}

/**
 * Update statistics display
 */
function updateStats() {
    const activeTracksEl = document.getElementById('activeTracks');
    const fpsEl = document.getElementById('fps');
    const totalDetectionsEl = document.getElementById('totalDetections');
    const memoryIdsEl = document.getElementById('memoryIds');
    
    if (!app.tracker || !app.isRunning) {
        if (activeTracksEl) activeTracksEl.textContent = '0';
        if (fpsEl) fpsEl.textContent = '0';
        if (totalDetectionsEl) totalDetectionsEl.textContent = '0';
        if (memoryIdsEl) memoryIdsEl.textContent = '0';
        return;
    }
    
    const currentTime = performance.now() / 1000;
    const tracks = app.tracker.getActiveTracks(currentTime);
    
    let activeCount = 0;
    let memoryCount = 0;
    
    for (const track of tracks) {
        if (track.state === 'active') {
            activeCount++;
        } else if (track.state === 'lost' && 
                   (currentTime - track.lastSeen) <= app.maxMemorySeconds) {
            memoryCount++;
        }
    }
    
    if (activeTracksEl) activeTracksEl.textContent = activeCount;
    if (fpsEl) fpsEl.textContent = app.fps;
    if (totalDetectionsEl) totalDetectionsEl.textContent = app.totalDetections;
    if (memoryIdsEl) memoryIdsEl.textContent = memoryCount;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
