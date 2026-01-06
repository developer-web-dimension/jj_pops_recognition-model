ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

        // Configuration
        const ONNX_MODEL_PATH = "/jimjam_fp16_2_classes_2.onnx"; 
        const CLASS_NAMES = ["Nothing", "jimjam"];
        const REQUIRED_STABLE_FRAMES = 3;

        const CLASS_THRESHOLDS = {
            jimjam: 0.92
        };

        // Eating Detection Constants
        const LIP_INDICES = [78,191,80,81,82,13,312,311,310, 178,88,95,402,318,324,308];
        const EAT_WINDOW_MS = 8000;
        const CHEW_TARGET = 1; // Number of chews needed to trigger "Eating Detected"
        const OPEN_THR = 0.08, CLOSE_THR = 0.04;
        const CONTACT_REQUIRED_MS = 100;
        const TOUCH_MIN_PX = 1;
        const TOUCH_MAX_PX = 60; // Max distance in pixels from lip center for tip to  
        //  register contact

        // State Variables
        let session;
        let inputName = null;
        let outputName = null;
        let latestFace = null;
        let latestHands = null;
        let gateRunning = false;
        let jimjamDetected = false;
        let jimjamTaken = false;
        let mouthState = "closed";
        let chewEvents = [];
        let eatingDetected = false;
        let contactStartTs = null;
        let lastHoldMs = 0;
        let holdingPrev = false;

        // ONNX Stability State
        let lastClass = null;
        let stableCount = 0;
        let currentJimjamClass = "Nothing";

        // DOM Elements
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const jimjamLabelEl = document.getElementById('jimjam-label');
        const jimjamStatusEl = document.getElementById('jimjam-status');
        const faceStatusEl = document.getElementById('face-status');
        const handStatusEl = document.getElementById('hand-status');
        const jimjamMouthStatusEl = document.getElementById('jimjam-mouth-status');
        const eatingStatusEl = document.getElementById('eating-status');
        const chewCountEl = document.getElementById('chew-count');
        const systemStatusEl = document.getElementById('system-status');
        const systemMessageEl = document.getElementById('system-message');
        
        // MediaPipe Setup
        const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
        faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        faceMesh.onResults(r => { latestFace = r; });

        const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
        hands.setOptions({ maxNumHands: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5, modelComplexity: 1 });
        hands.onResults(r => { latestHands = r; });
        

        // ---------- Utility Functions ----------

        function setStatus(el, text, cls) {
            if (!el) return;
            el.textContent = text;
            el.className = `status-value ${cls}`;
        }
        function getLipPoints(lm, w, h) {
            return LIP_INDICES.map(i => ({ x: Math.round(lm[i].x * w), y: Math.round(lm[i].y * h) }));
        }
        function euclid(a,b) { const dx=b.x-a.x, dy=b.y-a.y; return Math.hypot(dx,dy); }
        function center(pts){ let sx=0,sy=0; for(const p of pts){sx+=p.x; sy+=p.y;} return {x:sx/pts.length,y:sy/pts.length}; }

        function softmax(arr) {
            const max = Math.max(...arr);
            const exps = arr.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sum);
        }

        // ---------- ONNX Model Loading and Preprocessing ----------

        async function loadONNXModel() {
            setStatus(systemStatusEl, "Loading ONNX Model...", "warning");
            try {
                session = await ort.InferenceSession.create(ONNX_MODEL_PATH, {
                    executionProviders: ["wasm"],
                });

                // Detect actual input & output names
                inputName = session.inputNames[0];
                outputName = session.outputNames[0];

                console.log("ONNX Model Loaded. Input:", inputName, "Output:", outputName);
                jimjamLabelEl.innerText = "Ready to start 🎥";
                setStatus(systemStatusEl, "Ready", "pass");
            } catch (error) {
                const msg = "ONNX Model Load Error: " + error.message;
                console.error(msg, error);
                jimjamLabelEl.innerText = "Model Failed to Load ❌";
                setStatus(systemStatusEl, "Error", "fail");
                systemMessageEl.textContent = "Check if ONNX file is accessible: " + error.message;
                throw error;
            }
        }

        function preprocessForONNX() {
            // Use a temporary canvas context for preprocessing 
            // NOTE: Using the main canvas context temporarily for convenience.
            const w = 224, h = 224;
            ctx.drawImage(video, 0, 0, w, h);
            const data = ctx.getImageData(0, 0, w, h).data;

            const arr = new Float32Array(3 * w * h);
            let pixelIndex = 0;

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const i = (y * w + x) * 4;

                    const r = data[i] / 255;
                    const g = data[i + 1] / 255;
                    const b = data[i + 2] / 255;

                    // Normalize (ImageNet mean/std for common models)
                    arr[pixelIndex] = (r - 0.485) / 0.229;
                    arr[pixelIndex + w * h] = (g - 0.456) / 0.224;
                    arr[pixelIndex + 2 * w * h] = (b - 0.406) / 0.225;

                    pixelIndex++;
                }
            }
            // Clear the small draw
            ctx.clearRect(0, 0, w, h); 
            // Redraw video at full size
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);


            return new ort.Tensor("float32", arr, [1, 3, w, h]);
        }

        async function predictJimjam() {
            if (!session || jimjamDetected) return;

            try {
                const tensor = preprocessForONNX();
                const feeds = {};
                feeds[inputName] = tensor;

                const output = await session.run(feeds);
                const logits = output[outputName].data;

                // Apply softmax (convert logits → probabilities)
                const probs = softmax(logits);

                // Find top prediction
                let maxIdx = 0;
                let maxProb = probs[0];

                for (let i = 1; i < probs.length; i++) {
                    if (probs[i] > maxProb) {
                        maxProb = probs[i];
                        maxIdx = i;
                    }
                }

                console.log("ONNX Prediction:", CLASS_NAMES[maxIdx], maxProb.toFixed(3));

                const predictionClass = CLASS_NAMES[maxIdx];
                const percent = (maxProb * 100).toFixed(1);

                // Stability check
                if (maxIdx === lastClass) {
                    stableCount++;
                } else {
                    stableCount = 1;
                    lastClass = maxIdx;
                }
                
                // Update UI based on current prediction
                jimjamLabelEl.innerText = `${predictionClass} (${percent}%)`;
                
                if (stableCount >= REQUIRED_STABLE_FRAMES) {
                    currentJimjamClass = predictionClass;
                    const threshold = CLASS_THRESHOLDS[predictionClass] ?? 0.7;
                    const isJimjam = currentJimjamClass === "jimjam";

                    if (isJimjam && maxProb > threshold) {
                        jimjamDetected = true;
                        setStatus(jimjamStatusEl, `${currentJimjamClass} DETECTED ✅`, "pass");
                        jimjamLabelEl.style.backgroundColor = 'rgba(16, 185, 129, 0.8)'; // Green
                        console.log("JimJam Detected, activating eating tracker.");
                    } else {
                        // Not stable or not high confidence Jimjam
                        setStatus(jimjamStatusEl, `Searching... (${currentJimjamClass} ${percent}%)`, "warning");
                        jimjamLabelEl.style.backgroundColor = 'rgba(245, 158, 11, 0.8)'; // Orange
                    }
                } else {
                     setStatus(jimjamStatusEl, `Searching... (Stable ${stableCount}/${REQUIRED_STABLE_FRAMES})`, "warning");
                }


            } catch (e) {
                console.warn("ONNX Prediction failed:", e);
            }
        }

        function pushChew(ts){
            chewEvents.push(ts);
            const cut = ts - EAT_WINDOW_MS;
            while (chewEvents.length && chewEvents[0] < cut) chewEvents.shift();
            
            const was = eatingDetected;
            eatingDetected = chewEvents.length >= CHEW_TARGET;
            
            if (!was && eatingDetected) {
                console.log("EATING DETECTED ✅");
                setStatus(eatingStatusEl, "EATING ✅", "pass");
            }
        }

        function updateStatusPanel() {
            chewCountEl.textContent = chewEvents.length;

            if (eatingDetected) {
                setStatus(eatingStatusEl, "EATING ✅", "pass");
            } else {
                const baseText = jimjamDetected ? (jimjamTaken ? "Ready to detect chewing" : "Bring JimJam to mouth") : "Waiting for JimJam";
                setStatus(eatingStatusEl, baseText, jimjamDetected ? "warning" : "fail");
            }

            if (jimjamTaken) {
                setStatus(jimjamMouthStatusEl, "TAKEN TO MOUTH ✅", "pass");
            } else if (jimjamDetected) {
                setStatus(jimjamMouthStatusEl, "Bring to your mouth", "warning");
            } else {
                setStatus(jimjamMouthStatusEl, "Waiting for JimJam detection", "fail");
            }
            
            if (jimjamDetected) {
                setStatus(systemStatusEl, "Monitoring Chewing", "info");
            } else {
                setStatus(systemStatusEl, "Detecting JimJam", "warning");
            }
        }

        // ---- the per-frame loop ----
        async function onFrameLoop() {
            if (!gateRunning) return;

            // 1. Keep canvas synced with video dimensions
            if (video.videoWidth && video.videoHeight) {
                if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                }
            }

            // 2. Jimjam detection until seen
            if (!jimjamDetected) {
                await predictJimjam();
            }

            // 3. After Jimjam seen, run Face/Hands for eating detection
            if (jimjamDetected) {
                await faceMesh.send({ image: video });
                await hands.send({ image: video });
                await pose.send({ image: video });
            }

            // --- Drawing and Logic ---
            
            // Base frame
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const faceOk = jimjamDetected ? !!latestFace?.multiFaceLandmarks?.length : false;
            const handOk = jimjamDetected ? !!latestHands?.multiHandLandmarks?.length : false;

            faceStatusEl.textContent = faceOk ? "✅" : "❌";
            handStatusEl.textContent = handOk ? "✅" : "❌";

            // lip center + openness + chew FSM
            let lipCenter = null, openness = 0;
            if (faceOk && jimjamDetected) {
                const lm = latestFace.multiFaceLandmarks[0];
                const lips = getLipPoints(lm, canvas.width, canvas.height);
                lipCenter = center(lips);
                

                const pUp   = { x: lm[13].x * canvas.width,  y: lm[13].y * canvas.height };
                const pLow  = { x: lm[14].x * canvas.width,  y: lm[14].y * canvas.height };
                const pLeft = { x: lm[61].x * canvas.width,  y: lm[61].y * canvas.height };
                const pRgt  = { x: lm[291].x* canvas.width,  y: lm[291].y* canvas.height };
                const mouthW = Math.max(1, euclid(pLeft, pRgt));
                const mouthH = euclid(pUp, pLow);
                openness = mouthH / mouthW; // Ratio of height to width

                const ts = performance.now();
                if (mouthState === "closed" && openness > OPEN_THR) mouthState = "open";
                else if (mouthState === "open" && openness < CLOSE_THR) {
                    mouthState = "closed";
                    pushChew(ts);
                }
            }

            // hand→mouth hold logic
            if (handOk && lipCenter && jimjamDetected) {
                const h = latestHands.multiHandLandmarks[0];
                const indexTip = { x: h[8].x * canvas.width,  y: h[8].y * canvas.height };
                const thumbTip = { x: h[4].x * canvas.width,  y: h[4].y * canvas.height };
                const dI = euclid(lipCenter, indexTip);
                const dT = euclid(lipCenter, thumbTip);
                const inI = (dI >= TOUCH_MIN_PX && dI <= TOUCH_MAX_PX);
                const inT = (dT >= TOUCH_MIN_PX && dT <= TOUCH_MAX_PX);
                const holding = inI && inT;


                const now = performance.now();
                if (!jimjamTaken) {
                    if (holding) {
                        holdingPrev = true;
                        if (contactStartTs == null) contactStartTs = now;
                        lastHoldMs = now - contactStartTs;
                        if (lastHoldMs >= CONTACT_REQUIRED_MS) {
                            jimjamTaken = true;
                            console.log("JIMJAM TAKEN TO MOUTH ✅");
                        }
                    } else {
                        holdingPrev = false; contactStartTs = null; lastHoldMs = 0;
                    }
                }
                
                // UI Debugging Text
                ctx.fillStyle = "#fff"; ctx.font = "14px Inter";
                if (!jimjamTaken && holding) {
                    const secs = Math.min(CONTACT_REQUIRED_MS, lastHoldMs) / 1000;
                    ctx.fillStyle = "#ffd24d"; ctx.font = "bold 16px Inter";
                    ctx.fillText(`Hold near lips: ${secs.toFixed(1)} / ${(CONTACT_REQUIRED_MS/1000).toFixed(1)} s`, 10, 60);
                }
            } else {
                contactStartTs = null; lastHoldMs = 0; holdingPrev = false;
            }

            // Status instructions
            ctx.fillStyle = "#fff"; ctx.font = "bold 16px Inter";
            if (!jimjamDetected) {
                ctx.fillText("Step 1: Show JimJam to the camera", 10, 40);
            } else if (!jimjamTaken) {
                ctx.fillStyle = "#ffd24d";
                ctx.fillText("Step 2: Bring JimJam to your mouth", 10, 40);
            } else if (!eatingDetected) {
                ctx.fillStyle = "#ffd24d";
                ctx.fillText("Step 3: Start eating the JimJam", 10, 40);
            }
            
            // Final success message
            if (jimjamDetected && jimjamTaken && eatingDetected) {
                ctx.fillStyle = "#7cff8e";
                ctx.fillText("SUCCESS! Eating detected.", 10, 80);
            }

            updateStatusPanel();
            
            // Loop for next frame
            requestAnimationFrame(onFrameLoop);
        }

        // ---------- Initialization and Start ----------
        
        async function startCameraAndGate() {
            try {
                // 1. Load ONNX Model
                await loadONNXModel();
                
                // 2. Start Camera
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30, max: 30 }
                    },
                    audio: false
                    });

                video.srcObject = stream;
                
                // Set canvas size to camera resolution when video loads
                video.onloadedmetadata = () => {
                    video.width = video.videoWidth;
                    video.height = video.videoHeight;
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    video.setAttribute("playsinline", "");
                    video.setAttribute("webkit-playsinline", "");
                    video.muted = true;
                    video.autoplay = true;

                    video.play();
                    
                    // 3. Start the main processing loop
                    gateRunning = true;
                    requestAnimationFrame(onFrameLoop);
                };
            } catch (err) {
                jimjamLabelEl.innerText = "Camera Access Denied ❌";
                setStatus(systemStatusEl, "FATAL ERROR", "fail");
                systemMessageEl.textContent = "Could not start camera. Check permissions: " + err.message;
                console.error(err);
            }
        }
        
        window.toggleStatusPanel = function toggleStatusPanel() {
            document.getElementById('statusPanel').classList.toggle('minimized');
        };

        // Initialize on load
        startCameraAndGate();