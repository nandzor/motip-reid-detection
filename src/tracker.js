/**
 * Re-Identification Person Tracker with 60-Second Memory
 * Maintains person IDs even when they disappear from camera for up to 60 seconds
 */

export class PersonTracker {
    constructor(maxMemorySeconds = 60) {
        this.maxMemorySeconds = maxMemorySeconds;
        this.tracks = new Map(); // trackId -> Track
        this.nextId = 1;
        this.featureHistory = new Map(); // trackId -> feature history
        
        // Re-identification parameters for better accuracy
        this.minReIdDelaySeconds = 0.5; // Minimum time before re-ID (prevents immediate false matches)
        this.reIdSimilarityThreshold = 0.75; // Higher threshold for re-ID (was 0.5)
        this.maxReIdDistanceRatio = 2.0; // Max distance ratio from last position (2x bbox size)
    }

    /**
     * Extract simple feature vector from bounding box region
     * In production, this would use a proper Re-ID model
     */
    extractFeature(imageData, box, canvas) {
        const [x1, y1, x2, y2] = box.map(v => Math.round(v));
        
        // Get region of interest
        const ctx = canvas.getContext('2d');
        const width = x2 - x1;
        const height = y2 - y1;
        
        if (width <= 0 || height <= 0) {
            return null;
        }
        
        // Create temporary canvas for ROI
        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = width;
        roiCanvas.height = height;
        const roiCtx = roiCanvas.getContext('2d');
        
        // Draw cropped region
        roiCtx.drawImage(
            imageData,
            x1, y1, width, height,
            0, 0, width, height
        );
        
        // Get image data
        const roiData = roiCtx.getImageData(0, 0, width, height);
        
        // Extract simple features (histogram + spatial features)
        const features = this.computeFeatures(roiData);
        
        return features;
    }

    /**
     * Compute simple feature vector from image data
     * This is a simplified version - in production, use a proper Re-ID model
     */
    computeFeatures(imageData) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        
        // Color histogram features (RGB)
        const histR = new Array(8).fill(0);
        const histG = new Array(8).fill(0);
        const histB = new Array(8).fill(0);
        
        // Spatial features (center region vs edges)
        let centerR = 0, centerG = 0, centerB = 0;
        let edgeR = 0, edgeG = 0, edgeB = 0;
        let centerCount = 0, edgeCount = 0;
        
        const centerStartX = Math.floor(width * 0.25);
        const centerEndX = Math.floor(width * 0.75);
        const centerStartY = Math.floor(height * 0.25);
        const centerEndY = Math.floor(height * 0.75);
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            const x = (i / 4) % width;
            const y = Math.floor((i / 4) / width);
            
            // Histogram
            histR[Math.floor(r / 32)]++;
            histG[Math.floor(g / 32)]++;
            histB[Math.floor(b / 32)]++;
            
            // Spatial features
            if (x >= centerStartX && x < centerEndX && y >= centerStartY && y < centerEndY) {
                centerR += r;
                centerG += g;
                centerB += b;
                centerCount++;
            } else {
                edgeR += r;
                edgeG += g;
                edgeB += b;
                edgeCount++;
            }
        }
        
        // Normalize histogram
        const totalPixels = width * height;
        for (let i = 0; i < 8; i++) {
            histR[i] /= totalPixels;
            histG[i] /= totalPixels;
            histB[i] /= totalPixels;
        }
        
        // Normalize spatial features
        if (centerCount > 0) {
            centerR /= centerCount;
            centerG /= centerCount;
            centerB /= centerCount;
        }
        if (edgeCount > 0) {
            edgeR /= edgeCount;
            edgeG /= edgeCount;
            edgeB /= edgeCount;
        }
        
        // Combine features
        const features = [
            ...histR,
            ...histG,
            ...histB,
            centerR / 255.0,
            centerG / 255.0,
            centerB / 255.0,
            edgeR / 255.0,
            edgeG / 255.0,
            edgeB / 255.0,
            width / 1000.0, // normalized width
            height / 1000.0 // normalized height
        ];
        
        return new Float32Array(features);
    }

    /**
     * Calculate cosine similarity between two feature vectors
     */
    cosineSimilarity(feat1, feat2) {
        if (!feat1 || !feat2) return 0;
        
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < feat1.length; i++) {
            dotProduct += feat1[i] * feat2[i];
            norm1 += feat1[i] * feat1[i];
            norm2 += feat2[i] * feat2[i];
        }
        
        const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator > 0 ? dotProduct / denominator : 0;
    }

    /**
     * Calculate IoU between two bounding boxes
     */
    calculateIoU(box1, box2) {
        const [x1_min, y1_min, x1_max, y1_max] = box1;
        const [x2_min, y2_min, x2_max, y2_max] = box2;
        
        const interXMin = Math.max(x1_min, x2_min);
        const interYMin = Math.max(y1_min, y2_min);
        const interXMax = Math.min(x1_max, x2_max);
        const interYMax = Math.min(y1_max, y2_max);
        
        const interArea = Math.max(0, interXMax - interXMin) * Math.max(0, interYMax - interYMin);
        const box1Area = (x1_max - x1_min) * (y1_max - y1_min);
        const box2Area = (x2_max - x2_min) * (y2_max - y2_min);
        
        const unionArea = box1Area + box2Area - interArea;
        
        return unionArea > 0 ? interArea / unionArea : 0;
    }

    /**
     * Update tracker with new detections
     */
    update(detections, imageData, canvas, currentTime) {
        // Cleanup old tracks (older than maxMemorySeconds)
        this.cleanupOldTracks(currentTime);
        
        // Extract features for new detections
        const detectionFeatures = detections.map(det => ({
            ...det,
            feature: this.extractFeature(imageData, det.box, canvas)
        }));
        
        // Match detections to existing tracks
        const matchedPairs = [];
        const matchedTrackIds = new Set();
        const matchedDetectionIndices = new Set();
        
        // First pass: IoU matching for active tracks
        for (const [trackId, track] of this.tracks.entries()) {
            if (track.state !== 'active') continue;
            
            let bestIoU = 0;
            let bestDetIdx = -1;
            
            for (let i = 0; i < detectionFeatures.length; i++) {
                if (matchedDetectionIndices.has(i)) continue;
                
                const iou = this.calculateIoU(track.box, detectionFeatures[i].box);
                if (iou > bestIoU && iou > 0.3) {
                    bestIoU = iou;
                    bestDetIdx = i;
                }
            }
            
            if (bestDetIdx >= 0) {
                matchedPairs.push({ trackId, detIdx: bestDetIdx, matchType: 'iou', score: bestIoU });
                matchedTrackIds.add(trackId);
                matchedDetectionIndices.add(bestDetIdx);
            }
        }
        
        // Second pass: Re-ID matching for lost tracks (within memory window)
        // Collect all potential re-ID matches first, then select best matches
        const reIdCandidates = [];
        
        for (const [trackId, track] of this.tracks.entries()) {
            if (track.state !== 'lost' || matchedTrackIds.has(trackId)) continue;
            if (currentTime - track.lastSeen > this.maxMemorySeconds) continue;
            
            // Minimum time constraint: person must be lost for at least minReIdDelaySeconds
            const timeSinceLost = currentTime - track.lastSeen;
            if (timeSinceLost < this.minReIdDelaySeconds) continue;
            
            // Get average feature from track history
            const trackFeatures = this.featureHistory.get(trackId);
            if (!trackFeatures || trackFeatures.length === 0) continue;
            
            const avgFeature = this.averageFeatures(trackFeatures);
            
            // Find best match for this lost track
            let bestSimilarity = this.reIdSimilarityThreshold;
            let bestDetIdx = -1;
            let bestDistanceScore = 0;
            let bestCombinedScore = 0;
            
            for (let i = 0; i < detectionFeatures.length; i++) {
                if (matchedDetectionIndices.has(i)) continue;
                
                const detFeature = detectionFeatures[i].feature;
                if (!detFeature) continue;
                
                const similarity = this.cosineSimilarity(avgFeature, detFeature);
                if (similarity < this.reIdSimilarityThreshold) continue;
                
                // Spatial validation: check if detection is in reasonable distance from last position
                const [lastX1, lastY1, lastX2, lastY2] = track.box;
                const [detX1, detY1, detX2, detY2] = detectionFeatures[i].box;
                
                // Calculate centers
                const lastCenterX = (lastX1 + lastX2) / 2;
                const lastCenterY = (lastY1 + lastY2) / 2;
                const detCenterX = (detX1 + detX2) / 2;
                const detCenterY = (detY1 + detY2) / 2;
                
                // Calculate distance
                const dx = detCenterX - lastCenterX;
                const dy = detCenterY - lastCenterY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Calculate reference distance (bbox diagonal)
                const lastWidth = lastX2 - lastX1;
                const lastHeight = lastY2 - lastY1;
                const bboxDiagonal = Math.sqrt(lastWidth * lastWidth + lastHeight * lastHeight);
                const maxAllowedDistance = bboxDiagonal * this.maxReIdDistanceRatio;
                
                // Spatial constraint: must be within reasonable distance
                if (distance > maxAllowedDistance) continue;
                
                // Distance score (closer is better, normalized)
                const distanceScore = 1.0 - Math.min(1.0, distance / maxAllowedDistance);
                
                // Combined score: similarity weighted higher than distance
                const combinedScore = similarity * 0.8 + distanceScore * 0.2;
                
                // Update best match if combined score is better
                if (combinedScore > bestCombinedScore || 
                    (combinedScore === bestCombinedScore && similarity > bestSimilarity)) {
                    bestSimilarity = similarity;
                    bestDistanceScore = distanceScore;
                    bestCombinedScore = combinedScore;
                    bestDetIdx = i;
                }
            }
            
            if (bestDetIdx >= 0) {
                reIdCandidates.push({ 
                    trackId, 
                    detIdx: bestDetIdx, 
                    matchType: 'reid', 
                    similarity: bestSimilarity,
                    distanceScore: bestDistanceScore,
                    combinedScore: bestSimilarity * 0.8 + bestDistanceScore * 0.2
                });
            }
        }
        
        // Sort candidates by combined score (best first) and match greedily
        reIdCandidates.sort((a, b) => b.combinedScore - a.combinedScore);
        
        for (const candidate of reIdCandidates) {
            // Skip if already matched
            if (matchedTrackIds.has(candidate.trackId) || 
                matchedDetectionIndices.has(candidate.detIdx)) {
                continue;
            }
            
            matchedPairs.push({
                trackId: candidate.trackId,
                detIdx: candidate.detIdx,
                matchType: candidate.matchType,
                score: candidate.similarity
            });
            matchedTrackIds.add(candidate.trackId);
            matchedDetectionIndices.add(candidate.detIdx);
        }
        
        // Update matched tracks
        for (const { trackId, detIdx } of matchedPairs) {
            const track = this.tracks.get(trackId);
            const detection = detectionFeatures[detIdx];
            
            track.box = detection.box;
            track.score = detection.score;
            track.lastSeen = currentTime;
            track.state = 'active';
            track.frameCount++;
            
            // Update feature history
            if (detection.feature) {
                if (!this.featureHistory.has(trackId)) {
                    this.featureHistory.set(trackId, []);
                }
                const features = this.featureHistory.get(trackId);
                features.push(detection.feature);
                // Keep only last 10 features
                if (features.length > 10) {
                    features.shift();
                }
            }
        }
        
        // Mark unmatched tracks as lost
        for (const [trackId, track] of this.tracks.entries()) {
            if (track.state === 'active' && !matchedTrackIds.has(trackId)) {
                track.state = 'lost';
            }
        }
        
        // Create new tracks for unmatched detections
        for (let i = 0; i < detectionFeatures.length; i++) {
            if (matchedDetectionIndices.has(i)) continue;
            
            const detection = detectionFeatures[i];
            const trackId = this.nextId++;
            
            const track = {
                id: trackId,
                box: detection.box,
                score: detection.score,
                state: 'active',
                created: currentTime,
                lastSeen: currentTime,
                frameCount: 1,
                color: this.generateColor(trackId)
            };
            
            this.tracks.set(trackId, track);
            
            // Initialize feature history
            if (detection.feature) {
                this.featureHistory.set(trackId, [detection.feature]);
            }
        }
        
        return this.getActiveTracks(currentTime);
    }

    /**
     * Get average feature from feature history
     */
    averageFeatures(features) {
        if (features.length === 0) return null;
        
        const avg = new Float32Array(features[0].length);
        
        for (const feature of features) {
            for (let i = 0; i < feature.length; i++) {
                avg[i] += feature[i];
            }
        }
        
        for (let i = 0; i < avg.length; i++) {
            avg[i] /= features.length;
        }
        
        return avg;
    }

    /**
     * Cleanup tracks that are older than maxMemorySeconds
     */
    cleanupOldTracks(currentTime) {
        const toDelete = [];
        
        for (const [trackId, track] of this.tracks.entries()) {
            if (track.state === 'lost' && currentTime - track.lastSeen > this.maxMemorySeconds) {
                toDelete.push(trackId);
            }
        }
        
        for (const trackId of toDelete) {
            this.tracks.delete(trackId);
            this.featureHistory.delete(trackId);
        }
    }

    /**
     * Get all active tracks (including lost tracks within memory window)
     */
    getActiveTracks(currentTime) {
        const active = [];
        
        for (const track of this.tracks.values()) {
            if (track.state === 'active' || 
                (track.state === 'lost' && currentTime - track.lastSeen <= this.maxMemorySeconds)) {
                active.push(track);
            }
        }
        
        return active;
    }

    /**
     * Generate unique color for track ID
     */
    generateColor(id) {
        // Generate consistent colors based on ID
        const hue = (id * 137.508) % 360; // Golden angle approximation
        return this.hslToRgb(hue / 360, 0.7, 0.5);
    }

    /**
     * Convert HSL to RGB
     */
    hslToRgb(h, s, l) {
        let r, g, b;
        
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        
        return [
            Math.round(r * 255),
            Math.round(g * 255),
            Math.round(b * 255)
        ];
    }

    /**
     * Reset tracker (clear all tracks)
     */
    reset() {
        this.tracks.clear();
        this.featureHistory.clear();
        this.nextId = 1;
    }
}
