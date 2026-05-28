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
    // Check if the browser supports the required MediaDevices API
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Browser Error: MediaDevices streaming layers are missing.");
        return;
    }

    try {
        // Disable the button during the initialization sequence
        startButton.disabled = true;
        startButton.innerText = "Connecting Feed...";

        // Request permission and capture raw hardware stream pipeline
        // This opens a browser prompt. Execution pauses here until the user accepts or denies.
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user" // Prioritize front-facing selfie camera
            },
            audio: false
        });

        // Bind the active camera stream pipeline directly to the video element's source object
        webcamElement.srcObject = stream;

        // Wait until the video metadata has fully loaded and is ready to execute
        // We wrap this lifecycle event in a Promise to safely block downstream actions until the feed is hot.
        await new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                // The browser now knows the exact resolution and attributes of the incoming video feed
                resolve();
            };
        });

        // Play the video stream inside the element
        await webcamElement.play();

        // --- NEW: Phase 6 Buffer Sync ---
        // Force the invisible canvas coordinate system to perfectly map 1:1 to the raw video resolution
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        // --------------------------------

        // Update UI
        startButton.innerText = "Face Detection Active";

        // Trigger the asynchronous recursive object tracking execution loop
        // This creates a loop that runs: ~60 times per second (like animation)
        requestAnimationFrame(predictLoop);

    } catch (error) {
        // 4. Robust Error Handling Pipeline
        console.error("Camera Init Error Structure:", error);
        
        // Reset button state so user can retry if it was a transient error
        startButton.disabled = false;
        startButton.innerText = "Start Camera";

        // Evaluate specific error signatures thrown by the browser API
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            alert("Permission Denied: Please allow webcam access in your browser settings to run face detection.");
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            alert("Hardware Error: No compatible camera device could be found on your system.");
        } else {
            alert(`Stream Initialization Failed: ${error.message}`);
        }
    }
}

/**
 * Phase C & D: Infinite Frame Engine & Rendering Pipeline
 */
/**
 * The core render loop: Ingests a frame from the webcam,
 * runs the ML model, and draws bounding boxes on the canvas.
 */
function predictLoop() {
    // Only proceed if the stream is live and the ML engine is ready
    if (webcamElement.currentTime !== 0 && faceDetectorEngine !== null) {
        
        // Clear the canvas from the previous frame to avoid 'ghosting' or artifacts
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Run inference on the current video frame
        const frameTimestampMs = performance.now();
        const results = faceDetectorEngine.detectForVideo(webcamElement, frameTimestampMs);

        // Iterate through detected faces and render their bounding boxes
        if (results.detections?.length > 0) {
            results.detections.forEach((detection) => {
                const { originX, originY, width, height } = detection.boundingBox;

                // Set bounding box visual style
                canvasCtx.lineWidth = 4;
                canvasCtx.strokeStyle = '#2563eb';
                
                // Draw the rectangle based on raw model coordinate output
                canvasCtx.strokeRect(originX, originY, width, height);
            });
        }
    }

    // Schedule the next frame render before the next browser repaint
    requestAnimationFrame(predictLoop);
}

// System Init Hook
initializeFaceDetectorEngine();

// UI Event Listener
startButton.addEventListener('click', setupWebcam);