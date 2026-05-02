const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const uploadPrompt = document.getElementById('uploadPrompt');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnLoader = document.getElementById('btnLoader');
const resultsArea = document.getElementById('resultsArea');

// Handle Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        imageInput.files = e.dataTransfer.files;
        updateImagePreview();
    }
});

uploadArea.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', updateImagePreview);

function updateImagePreview() {
    if (imageInput.files && imageInput.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadPrompt.style.display = 'none';
            analyzeBtn.disabled = false;
            resultsArea.style.display = 'none'; // Hide old results
        };
        reader.readAsDataURL(imageInput.files[0]);
    }
}

analyzeBtn.addEventListener('click', async () => {
    if (!imageInput.files[0]) return;

    // UI Loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('span').textContent = 'Running Model...';
    btnLoader.style.display = 'block';
    resultsArea.style.display = 'none';

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Failed to run inference: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('span').textContent = 'Run Edge Inference';
        btnLoader.style.display = 'none';
    }
});

function displayResults(data) {
    resultsArea.style.display = 'block';
    
    // Update Prediction
    const predClass = document.getElementById('predClass');
    predClass.textContent = data.prediction;
    
    // Change color based on health
    if (data.prediction.toLowerCase() === 'healthy') {
        predClass.style.color = 'var(--success)';
        document.getElementById('predConfidenceBar').style.background = 'var(--success)';
    } else {
        predClass.style.color = 'var(--warning)';
        document.getElementById('predConfidenceBar').style.background = 'var(--warning)';
    }

    document.getElementById('predConfidence').textContent = `${data.confidence}% Confidence`;
    setTimeout(() => {
        document.getElementById('predConfidenceBar').style.width = `${data.confidence}%`;
    }, 100);

    // Update Metrics
    document.getElementById('valLatency').innerHTML = `${data.latency_ms}<small>ms</small>`;
    document.getElementById('valSize').innerHTML = `${data.model_size_mb}<small>MB</small>`;
    document.getElementById('valPower').textContent = data.power_score;

    // Status dots logic
    const statusLatency = document.getElementById('statusLatency');
    if (data.latency_ms <= 100) {
        statusLatency.className = 'status-dot green';
    } else {
        statusLatency.className = 'status-dot red';
    }

    const statusSize = document.getElementById('statusSize');
    if (data.model_size_mb <= 50) {
        statusSize.className = 'status-dot green';
    } else {
        statusSize.className = 'status-dot red';
    }
}
