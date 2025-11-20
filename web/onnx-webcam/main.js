import * as ort from "onnxruntime-web";

// Force wasm from CDN to avoid Vite MIME issues
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

const CLASS_NAMES = ["Nothing", "jimjam", "jimjam_pops"];
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const labelBox = document.getElementById("label-box");

let session;

async function loadModel() {
  labelBox.innerText = "Loading model...";
  session = await ort.InferenceSession.create("/jimjam_classifier.onnx", {
    executionProviders: ["wasm"]
  });
  labelBox.innerText = "Model Loaded ✔";
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();
    requestAnimationFrame(runLoop);
  };
}

function preprocess() {
  ctx.drawImage(video, 0, 0, 224, 224);
  const data = ctx.getImageData(0, 0, 224, 224).data;

  const arr = new Float32Array(3 * 224 * 224);

  let pixelIndex = 0;
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {

      const i = (y * 224 + x) * 4;

      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      // CHW format
      arr[pixelIndex] = (r - 0.485) / 0.229;                     // R channel
      arr[pixelIndex + 224 * 224] = (g - 0.456) / 0.224;         // G channel
      arr[pixelIndex + 2 * 224 * 224] = (b - 0.406) / 0.225;     // B channel

      pixelIndex++;
    }
  }

  return new ort.Tensor("float32", arr, [1, 3, 224, 224]);
}


async function runLoop() {
  const tensor = preprocess();
  const output = await session.run({ input: tensor });

  const result = output.output.data;
  const maxIdx = result.indexOf(Math.max(...result));

  labelBox.innerText = `${CLASS_NAMES[maxIdx]} (${result[maxIdx].toFixed(2)})`;

  requestAnimationFrame(runLoop);
}

await loadModel();
await startCamera();
