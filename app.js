/**
 * AI Face Detector Web Application
 * Core Machine Learning Pipeline Module using MediaPipe Tasks-Vision
 */

// 1. Explicitly import components from the MediaPipe ESM CDN
import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs";

// 2. Select DOM Viewport Elements
const webcamElement = document.getElementById('webcam');
const startButton = document.getElementById('start-camera-btn');

// Global reference pointer for our initialized ML Face Detection instantiation
let faceDetectorEngine = null;

/**
 * Phase A: Bootstrapping and compiling the ML Inference Brain
 */
async function initializeFaceDetectorEngine() {
    try {
        startButton.disabled = true;
        startButton.innerText = "Loading AI Model...";

        // Fetch WebAssembly (WASM) background processing assets 
        const wasmLocatorFileset = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );

        // Instantiate the detector backend with specific system settings
        faceDetectorEngine = await FaceDetector.createFromOptions(wasmLocatorFileset, {
            baseOptions: {
                // FIXED: Added missing underscores in the model filename "blaze_face_short_range"
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

        // Play the video stream inside the element & Update UI
        await webcamElement.play();
        startButton.innerText = "Face Detection Active";

        // Trigger the asynchronous recursive object tracking execution loop
        requestAnimationFrame(predictLoop);

    } catch (error) {
        // Robust Error Handling Pipeline
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
 * Phase C: Infinite Frame Engine - Processing Viewport Textures In Real-Time
 */
function predictLoop() {
    // Edge case: Safety check to confirm video stream element contains valid decoding buffers
    if (webcamElement.currentTime !== 0 && faceDetectorEngine !== null) {
        
        // Performance Timestamp marker needed by MediaPipe for processing live sequence records
        const frameTimestampMs = performance.now();

        // RUN INFERENCE CLIENT-SIDE: Pass raw video frame buffer snapshot directly to ML engine
        const inferenceOutputResult = faceDetectorEngine.detectForVideo(webcamElement, frameTimestampMs);

        // Parse and log resulting telemetry array to standard output
        if (inferenceOutputResult.detections && inferenceOutputResult.detections.length > 0) {
            console.clear(); // Keeps console neat by clearing previous frame logs
            console.log(`=== Frames Ingested Tracker: Found ${inferenceOutputResult.detections.length} Face(s) ===`);
            
            inferenceOutputResult.detections.forEach((detection, index) => {
                const boundingBox = detection.boundingBox;
                const score = (detection.categories[0].score * 100).toFixed(1);

                console.log(`Face #${index + 1} [Confidence: ${score}%]:`);
                console.log(`  -> Origin Offset X: ${boundingBox.originX.toFixed(2)}px`);
                console.log(`  -> Origin Offset Y: ${boundingBox.originY.toFixed(2)}px`);
                console.log(`  -> Render Width  : ${boundingBox.width.toFixed(2)}px`);
                console.log(`  -> Render Height : ${boundingBox.height.toFixed(2)}px`);
            });
        }
    }

    // Keep loop active: Request browser queue execution path again on subsequent monitor screen refresh
    requestAnimationFrame(predictLoop);
}

// System Init Hook: Bootstrap Model Assets immediately when DOM interpretation finishes loading
initializeFaceDetectorEngine();

// UI Event Listener: Route interaction path directly to camera hardware requests
startButton.addEventListener('click', setupWebcam);