/**
 * AI Face Detector Web Application
 * Core Machine Learning Pipeline Module using MediaPipe Tasks-Vision
 */

import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs";

// 1. Select DOM Viewport Elements
const webcamElement = document.getElementById('webcam');
const startButton = document.getElementById('start-camera-btn');

// --- NEW: Phase 6 Setup ---
// Capture the canvas element and instantiate its 2D drawing engine (context)
const canvasElement = document.getElementById('overlay');
const canvasCtx = canvasElement.getContext('2d');
// --------------------------

let faceDetectorEngine = null;

/**
 * Phase A: Bootstrapping and compiling the ML Inference Brain
 */
async function initializeFaceDetectorEngine() {
    try {
        startButton.disabled = true;
        startButton.innerText = "Loading AI Model...";

        const wasmLocatorFileset = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );

        faceDetectorEngine = await FaceDetector.createFromOptions(wasmLocatorFileset, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                delegate: "GPU" 
            },
            runningMode: "VIDEO", 
            minDetectionConfidence: 0.55 
        });

        startButton.disabled = false;
        startButton.innerText = "Start Camera";
        console.log("System Status: ML WebAssembly Inference Pipeline Compiled Successfully.");

    } catch (error) {
        console.error("Critical Failure Compiling Model Assets:", error);
        startButton.innerText = "Model Failed to Load";
    }
}

/**
 * Phase B: Establishing Hardware Access & Synchronizing Stream Input
 */
async function setupWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Browser Error: MediaDevices streaming layers are missing.");
        return;
    }

    try {
        startButton.disabled = true;
        startButton.innerText = "Connecting Feed...";

        const cameraHardwareTrack = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false
        });

        webcamElement.srcObject = cameraHardwareTrack;

        await new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => resolve();
        });

        await webcamElement.play();

        // --- NEW: Phase 6 Buffer Sync ---
        // Force the invisible canvas coordinate system to perfectly map 1:1 to the raw video resolution
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        // --------------------------------

        startButton.innerText = "Inference Engine Active";
        requestAnimationFrame(predictLoop);

    } catch (error) {
        console.error("Camera Binding Exception:", error);
        startButton.disabled = false;
        startButton.innerText = "Start Camera";
    }
}

/**
 * Phase C & D: Infinite Frame Engine & Rendering Pipeline
 */
/**
 * Phase C & D: Infinite Frame Engine & Rendering Pipeline
 */
function predictLoop() {
    if (webcamElement.currentTime !== 0 && faceDetectorEngine !== null) {
        
        // Wipe the canvas clean at the start of every frame. 
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        const frameTimestampMs = performance.now();
        const inferenceOutputResult = faceDetectorEngine.detectForVideo(webcamElement, frameTimestampMs);

        if (inferenceOutputResult.detections && inferenceOutputResult.detections.length > 0) {
            
            inferenceOutputResult.detections.forEach((detection) => {
                const boundingBox = detection.boundingBox;

                // --- MODIFIED: Phase 6 Vector Drawing (Box Only) ---
                // 1. Configure the brush styles for the bounding box
                canvasCtx.lineWidth = 4;
                canvasCtx.strokeStyle = '#2563eb'; // Blue matching your UI button
                
                // 2. Draw the rectangle using the raw pixel vectors from the ML model
                canvasCtx.strokeRect(
                    boundingBox.originX, 
                    boundingBox.originY, 
                    boundingBox.width, 
                    boundingBox.height
                );
                // -----------------------------------
            });
        }
    }

    requestAnimationFrame(predictLoop);
}

// System Init Hook
initializeFaceDetectorEngine();

// UI Event Listener
startButton.addEventListener('click', setupWebcam);