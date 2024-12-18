// Import TensorFlow.js library
import * as tf from '@tensorflow/tfjs';
import { loadImage } from '@tensorflow/tfjs-core';
import { createCanvas, ImageData } from 'canvas';

// Define the path to your TensorFlow Lite model and label text file
const MODEL_URL = 'model.tflite';
const LABELS_URL = 'labels.txt';

// Load the model and labels
let model;
let labels;

async function loadModel() {
    // Load the TensorFlow Lite model
    model = await tf.loadGraphModel(MODEL_URL);

    // Load the labels
    const response = await fetch(LABELS_URL);
    labels = await response.text();

    // Split the labels by newline to create an array
    labels = labels.split('\n');
}

// Function to preprocess the image for prediction
async function preprocessImage(image) {
    // Resize the image to match the model input size
    const resizedImage = tf.image.resizeBilinear(image, [224, 224]);

    // Convert the image to a tensor
    const tensor = tf.cast(resizedImage, 'float32');

    // Normalize the pixel values
    const offset = tf.scalar(127.5);
    const normalizedImage = tensor.sub(offset).div(offset);

    // Add a batch dimension
    const batchedImage = normalizedImage.expandDims(0);

    return batchedImage;
}

// Function to perform prediction
async function predict(imageElement) {
    // Convert the image to a tensor
    const imageTensor = tf.browser.fromPixels(imageElement);

    // Preprocess the image
    const preprocessedImage = await preprocessImage(imageTensor);

    // Perform prediction
    const prediction = model.predict(preprocessedImage);

    // Get the index of the predicted class
    const predictedClassIndex = prediction.argMax(1).dataSync()[0];

    // Get the predicted class label
    const predictedLabel = labels[predictedClassIndex];

    return predictedLabel;
}

// Function to handle webcam image capture and prediction
async function handleWebcamPrediction() {
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.createElement('canvas');
    const context = canvasElement.getContext('2d');

    // Set canvas size to match webcam video feed
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;

    // Draw webcam video frame on canvas
    context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);

    // Get image data from canvas
    const imageData = context.getImageData(0, 0, canvasElement.width, canvasElement.height);

    // Convert image data to TensorFlow.js tensor
    const tensor = tf.browser.fromPixels(imageData);

    // Perform prediction
    const predictedLabel = await predict(tensor);

    // Update UI with predicted label
    document.getElementById('predicted-text').innerText = predictedLabel;

    // Request animation frame to continuously capture webcam frames
    requestAnimationFrame(handleWebcamPrediction);
}

// Load the model and labels when the script is loaded
loadModel().then(() => {
    // Start webcam and perform prediction
    startWebcam();
});

// Function to start webcam and perform prediction
async function startWebcam() {
    const video = document.getElementById('webcam');
    try {
        // Get webcam access
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Start prediction loop
        handleWebcamPrediction();
    } catch (err) {
        console.error('Error accessing webcam:', err);
    }
}
