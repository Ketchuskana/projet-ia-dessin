async function run() {
  // const session = await ort.InferenceSession.create('./model/mnist_model.onnx');
  const session = await ort.InferenceSession.create(
    './model/mnist_model.onnx',
    { executionProviders: ['wasm'] }  // ou ['cpu']
  );

  const img = document.getElementById('mnist-img');
  
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0, 28, 28);

  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = imageData.data;

  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    input[i] = (255 - data[i * 4]) / 255;
  }

  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
  const feeds = { input: tensor };

  const results = await session.run(feeds);
  const output = results.output.data;

  let maxIndex = 0;
  for (let i = 1; i < output.length; i++) {
    if (output[i] > output[maxIndex]) maxIndex = i;
  }

  document.getElementById('output').textContent = `Prédiction: ${maxIndex}`;
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('runInference').addEventListener('click', run);
});
