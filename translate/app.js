//meta(not so imp) functions by vkc!

// const modelURL = "https://teachablemachine.withgoogle.com/models/QoG5AWCXa/model.json"; //working
//https://teachablemachine.withgoogle.com/models/rOKL3ianG/  //isl-BEST
//https://teachablemachine.withgoogle.com/models/nGWUB3-sm/  //ASL-BEST
const modelURL =
  "https://teachablemachine.withgoogle.com/models/rOKL3ianG/model.json";

// const metadataURL = "https://teachablemachine.withgoogle.com/models/QoG5AWCXa/metadata.json"; // working
const metadataURL =
  "https://teachablemachine.withgoogle.com/models/rOKL3ianG/metadata.json";

//https://teachablemachine.withgoogle.com/models/QoG5AWCXa/

let gttsBtn = document.getElementById("gtts-btn");
let micBtn = document.getElementById("mic-btn");
let keyboardBtn = document.getElementById("keyboard-btn");
let cameraBtn = document.getElementById("camera-btn");
let uploadBtn = document.querySelector("#upload-btn");
let textArea = document.querySelector("#textarea");
let backBtn = document.querySelector("#back-btn");
let line = document.createElement("p");
let bottomSheet = document.querySelector("#bottom-sheet");
let aiSuggestText = document.querySelector("#ai-suggest-text");

// set default dummy value in local storage

gttsBtn.style.background = "#193053";

line.className = "line";
line.id = "last-line";
line.innerHTML = `Ready to roll !`;
textArea.appendChild(line);
line.scrollIntoView({
  behavior: "smooth",
  block: "end",
  inline: "nearest",
});

// back btn
backBtn.addEventListener("click", function () {
  window.location = "../";
});

keyboardBtn.addEventListener("click", function () {
  window.location = "./text-to-sign";
});

let webcamON = false;

//open github repo
document.querySelector("#webcam-banner").addEventListener("click", function () {
  window.open("https://github.com/lumaticai/Sanket/", "_blank");
});

// test function
async function addTestLines(totalLines) {
  for (let i = 1; i < totalLines + 1; i++) {
    let line = document.createElement("p");
    line.className = "line";
    line.innerHTML = `Hello`;
    textArea.appendChild(line);
    line.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }
}
// addTestLines(10);

// adds new line in transcript box
async function addNewTranslateLine(text) {
  let lastLine = textArea.lastElementChild;
  try {
    document.querySelector("#last-line").id = "";
  } catch (err) {
    console.log(err);
  }

  let newLine = document.createElement("p");
  newLine.innerHTML = text;
  newLine.className = "line";
  newLine.id = "last-line";
  textArea.appendChild(newLine);
  newLine.scrollIntoView({
    behavior: "smooth",
    block: "end",
    inline: "nearest",
  });
}

// start sign prediction


// start sign prediction
// start sign prediction
cameraBtn.addEventListener("click", function () {
  window.location.href = "re-index.html"; // Replace "path/to/your/html/page.html" with the actual path to your HTML page
});


// text to speach btn colour change
// gttsBtn.addEventListener("click", function () {
//   if (gttsBtn.style.backgroundColor === "rgb(255,255,255)") {
//     gttsBtn.style.background = "#0DA778"; //dactivate btn
//     // alert('stopping... Press OK')
//     // location.reload()
//   } else {
//     gttsBtn.style.background = "#fff";
//     // alert('starting.... Press OK')
//     // init();
//   }
// });

// gttsBtn.addEventListener("click", function () {
//   if (gttsBtn.style.background === "rgb(25, 48, 83)") {
//     gttsBtn.style.background = "#62E6BF"; //active
//     console.log("gtts active");
//     // alert('stopping... Press OK')
//     // location.reload()
//   } else {
//     gttsBtn.style.background = "#193053";
//     console.log("gtts inactive");
//     // alert('starting.... Press OK')
//     // init();
//   }
// });

bottomSheet.addEventListener("click", function () {
  // get the last line and do gtts
  tts(aiSuggestText.innerText);
});

gttsBtn.addEventListener("click", async function () {
  if (gttsBtn.style.background === "rgb(25, 48, 83)") {
    gttsBtn.style.background = "#62E6BF"; // Activate button
    console.log("gtts active");

    // Retrieve the text from the last line
    const lastLineText = document.querySelector("#last-line").innerText;

    // Check if there is text to be spoken
    if (lastLineText.trim() !== "") {
      // Call the text-to-speech function
      await tts(lastLineText);
    } else {
      console.log("No text available for speech synthesis.");
    }
  } else {
    gttsBtn.style.background = "#193053"; // Deactivate button
    console.log("gtts inactive");
  }
});

// micBtn.addEventListener("click", function () {
//   if (micBtn.style.backgroundColor === "rgb(211, 241, 233)") {
//     micBtn.style.backgroundColor = "#62E6BF"; //active
//     console.log("mic active");
//   } else {
//     micBtn.style.backgroundColor = "rgb(211, 241, 233)";
//     console.log("mic not active");
//   }
// });
micBtn.addEventListener("click", function () {
  if (micBtn.style.backgroundColor === "rgb(211, 241, 233)") {
    micBtn.style.backgroundColor = "#62E6BF"; // Activate button
    console.log("mic active");

    // Start speech recognition
    startSpeechRecognition();
  } else {
    micBtn.style.backgroundColor = "rgb(211, 241, 233)"; // Deactivate button
    console.log("mic not active");

    // Stop speech recognition
    stopSpeechRecognition();
  }
});

let recognition;

async function startSpeechRecognition() {
  // Check if SpeechRecognition is supported
  if ("webkitSpeechRecognition" in window) {
    // Initialize speech recognition
    recognition = new webkitSpeechRecognition();
    recognition.continuous = true; // Enable continuous speech recognition
    recognition.lang = "en-US"; // Set language to English (United States)

    // Event listener for speech recognition result
    recognition.onresult = function (event) {
      const transcript = event.results[event.results.length - 1][0].transcript;

      // local storage data fetch
      const storedDetails = localStorage.getItem("details");
      if (storedDetails) {
        console.log("✨ asking gemini: " + transcript);
        geminiAIChat(`I am providing you with my personal details, and a query give the answer of the query in first person language only if it is possible from the personal info provided, if you cannot generate any answer just say "..." 
            Query:
            "${transcript}"
            
            Personal Info:
            "${storedDetails}"`);
      }

      //   console.log(transcript);
      // Add the transcript to the transcript box
      addNewTranslateLine("🗣️ " + transcript);
    };

    // Start speech recognition
    recognition.start();
  } else {
    console.error("SpeechRecognition is not supported in this browser.");
  }
}

function stopSpeechRecognition() {
  // Stop speech recognition if it's running
  if (recognition) {
    recognition.stop();
  }
}

///////////////////////////////////////////////
/////////// GEMINI MAGIC ////////////////////
//////////////////////////////////////////////
import { GoogleGenerativeAI } from "@google/generative-ai";

const magic = atob("QUl6YVN5QWI0ZzE5ai1SeXIzekdWTXNvZlZqalZQV3IyTlNZZnVn");
const genAI = new GoogleGenerativeAI(magic);

async function geminiAIChat(prompt) {
  // For text-only input, use the gemini-pro model
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const result = await model.generateContent(prompt);
  const response = await result.response;
  const text = response.text();
  aiSuggestText.innerText = text;
  console.log(text);
}

let file;

async function predictImage(file) {
  const modelURL = "https://teachablemachine.withgoogle.com/models/riOJKjJZ1/";
  const model = await tmImage.load(
    modelURL + "model.json",
    modelURL + "metadata.json"
  );

  const imagePreview = document.getElementsByClassName("webcam-view");

  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = async () => {
    imagePreview.src = reader.result;
    const imageElement = document.createElement("img");
    imageElement.src = reader.result;
    const prediction = await model.predict(imageElement);
    addNewTranslateLine(predictions);
    console.log(prediction);
  };
}

//upload img
// uploadBtn.addEventListener("click", function () {
//   uploadBtn.style.background = "#62E6BF";
//   uploadBtn.style.background = "";
//   const fileInput = document.querySelector("#file-input");
//   fileInput.click();
//   fileInput.addEventListener("change", async (event) => {
//     file = event.target.files[0];
//     if (!file) {
//       console.error("No file selected.");
//       return;
//     }
//   });
//   predictImage(file);
// let fileInput = document.querySelector('#file-input')
// fileInput.click();
// fileInput.addEventListener('change', async(event) => {
//     const file = event.target.files[0];
//     if (!file) {
//         console.error('No file selected.');
//         return;
//     }

//     const reader = new FileReader();
//     reader.readAsDataURL(file);
//     reader.onload = async() => {
//         let img = document.createElement('img')
//         img.className = 'webcam-view'
//         document.querySelector(".webcam-view").appendChild(img).src = reader.result;

//         //predict image

//         model = window.tmImage.load(modelURL, metadataURL);
//         let flip = true;
//         // maxPredictions = model.getTotalClasses();
//         const prediction = await model.predict(img);
//         console.log(prediction)
//         addNewTranslateLine(prediction)

//     };
//     reader.onerror = (error) => {
//         console.error(error);
//     };
// });
// });

//text to speech
async function tts(text) {
  if ("speechSynthesis" in window) {
    // Speech Synthesis is supported 🎉
    console.log("");
  } else {
    alert("Text to speech not available 😞");
    location.reload();
  }

  let msg = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(msg);
}

// delay function
const delay = (ms) => new Promise((res) => setTimeout(res, ms));

/////////////////////////////////////
let index = 0;
// const spamFilter = ["Hello","HOW ARE YOU ?","I NEED HELP","MY","NAME","IS","J","E","T","WHAT IS THE TIME ?","THANK YOU"]
let beforeTextDone = 0;

/////////////////////////////////////
// tensorflow.js magic /////////////
////////////////////////////////////
let model, webcam, ctox, labelContainer, maxPredictions;
async function init() {
  //hello, welcome, thankyou, iloveu {>= 0.95}
  // const modelURL = "https://storage.googleapis.com/tm-model/C2gYk6JPd/model.json";
  // const metadataURL = "https://storage.googleapis.com/tm-model/C2gYk6JPd/metadata.json";

  //1,2,3,4,5{==1.0 meh}
  // const modelURL = "https://storage.googleapis.com/tm-model/qWNVsgTyJ/model.json";
  // const metadataURL = "https://storage.googleapis.com/tm-model/qWNVsgTyJ/metadata.json";

  // load the model and metadata
  // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
  // or files from your local hard drive
  // Note: the pose library adds "tmImage" object to your window (window.tmImage)
  model = await tmImage.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Convenience function to setup a webcam
  const flip = true; // whether to flip the webcam
  webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip 180
  await webcam.setup(); // request access to the webcam
  await webcam.play();
  window.requestAnimationFrame(loop);

  // append elements to the DOM
  document.querySelector(".webcam-view").appendChild(webcam.canvas).className =
    "canvas";
  labelContainer = document.getElementById("last-line");
  for (let i = 0; i < maxPredictions; i++) {
    // and class labels
    labelContainer.appendChild(document.createElement("div"));
  }
}

async function loop() {
  webcam.update(); // update the webcam frame
  await predict();
  window.requestAnimationFrame(loop);
}
let Ebool = true;
// run the webcam image through the image model
async function predict() {
  // predict can take in an image, video or canvas html element
  const prediction = await model.predict(webcam.canvas);
  for (let i = 0; i < maxPredictions; i++) {
    // if (prediction[i].probability.toFixed(2) == 1.00) {
    //     console.log(prediction[i].probability.toFixed(2), prediction[i].className)
    // }
    // const classPrediction =  prediction[i].className + ": " + prediction[i].probability.toFixed(2);
    console.log(prediction[i].probability.toFixed(2), prediction[i].className);
    if (prediction[i].probability.toFixed(2) >= 0.9) {
      //     if (Ebool && document.querySelector('#last-line').innerHTML == "E" && prediction[i].className=="E"){
      //         Ebool=false;
      //         setTimeout(function() {addNewTranslateLine("E")}, 2000);

      //         // alert(11);

      //         if (gttsBtn.style.backgroundColor === 'rgb(212, 236, 126)') { //btn active
      //             await tts("E")
      //                 // delay(0)

      //         } else {
      //             console.log('')
      //         }
      //     }
      // if (prediction[i].className == spamFilter[index]) {
      if (1) {
        index += 1;
        // if (index == spamFilter.length) {
        //     index = 0;
        // }

        if (
          document.querySelector("#last-line").innerHTML !=
          prediction[i].className
        ) {
          // addNewTranslateLine(prediction[i].className);
          if (prediction[i].className === "IGNORE") {
            let ignore = "";
          } else {
            await addNewTranslateLine(prediction[i].className);
            if (gttsBtn.style.backgroundColor === "rgb(98, 230, 191)") {
              //btn active
              await tts(prediction[i].className);
              // delay(0)
            } else {
              console.log("");
            }
          }
        }
      }

      // labelContainer.childNodes[i].innerHTML = prediction[i].className;
    }
  }

  // for (let i = 0; i < maxPredictions; i++) {
  //     const classPrediction =
  //         prediction[i].className + ": " + prediction[i].probability.toFixed(2);
  //     labelContainer.childNodes[i].innerHTML = classPrediction;
  // }
}
