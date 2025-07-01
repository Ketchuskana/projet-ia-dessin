window.onload = () => {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const resultEl = document.getElementById('result');
  // const predictBtn = document.getElementById('predictBtn');
  const predictBtn = document.getElementById('predictBtn');
  predictBtn.onclick = predict;
  const clearBtn = document.getElementById('clearBtn');
  
  let isDrawing = false;
  
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  
  canvas.addEventListener('mousedown', () => isDrawing = true);
  canvas.addEventListener('mouseup', () => isDrawing = false);
  canvas.addEventListener('mouseleave', () => isDrawing = false);
  
  canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
  
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  });
  
  // Nettoyer canvas
  clearBtn.onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultEl.textContent = "Résultat : ?";
  };

  async function preprocess() {
    // Crée un canvas intermédiaire 28x28
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext('2d');
  
    // Dessine le contenu de grand canvas dans petit, en réduisant la taille
    smallCtx.drawImage(canvas, 0, 0, 28, 28);
  
    // Récupère les données pixels (RGBA)
    const imgData = smallCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;
  
    const input = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const r = data[i * 4];
      input[i] = (255 - r) / 255;
    }
    return input;
  }
  
  async function predict() {
    const input = await preprocess();
  
    // Créer session ONNX
    const session = await ort.InferenceSession.create(
      '/projet-ia-dessin/model/mnist_model.onnx',
      { executionProviders: ['wasm'] } 
    );
    const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
    const feeds = { input: tensor };
  
    const results = await session.run(feeds);
  
    const outputData = results.output.data;

    let maxIdx = 0;
    let maxVal = outputData[0];
    for (let i = 1; i < outputData.length; i++) {
      if (outputData[i] > maxVal) {
        maxVal = outputData[i];
        maxIdx = i;
      }
    }
  
    resultEl.textContent = `Résultat : ${maxIdx}`;
  }
};  

  

predictBtn.onclick = predict;

