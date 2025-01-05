// import Chart from 'chart.js/auto';


// Get the canvas context
const context = document.getElementById("mycanvas").getContext("2d");

// Canvas styling
context.strokeStyle = "#000000";
context.lineJoin = "round";
context.lineWidth = 8;

// Variables for drawing
let isPainting = false;
let clickX = [];
let clickY = [];
let clickDrag = [];

// Mouse event handlers
$("#mycanvas").mousedown((e) => {
  const mouseX = e.pageX - e.target.offsetLeft;
  const mouseY = e.pageY - e.target.offsetTop;

  isPainting = true;
  addClick(mouseX, mouseY);
  drawCanvas();
});

$("#mycanvas").mousemove((e) => {
  if (isPainting) {
    const mouseX = e.pageX - e.target.offsetLeft;
    const mouseY = e.pageY - e.target.offsetTop;

    addClick(mouseX, mouseY, true);
    drawCanvas();
  }
});

$("#mycanvas").mouseup(() => (isPainting = false));
$("#mycanvas").mouseleave(() => (isPainting = false));

// Add drawing coordinates
function addClick(x, y, dragging = false) {
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

// Draw on the canvas
function drawCanvas() {
  context.clearRect(0, 0, 200, 200); // Clear canvas

  for (let i = 0; i < clickX.length; i++) {
    context.beginPath();
    if (clickDrag[i] && i) {
      context.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
      context.moveTo(clickX[i] - 1, clickY[i]);
    }
    context.lineTo(clickX[i], clickY[i]);
    context.closePath();
    context.stroke();
  }
}

// Get pixel data
function getPixels() {
  const rawPixels = context.getImageData(0, 0, 200, 200).data;
  return Array.from({ length: rawPixels.length / 4 }, (_, i) => rawPixels[i * 4 + 3]);
}

// Submit pixel data
async function submitPixels() {
  const pixels = getPixels();

  try {
    const response = await fetch("http://127.0.0.1:5000/print_pixels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pixels),
    });

    if (response.ok) {
      const result = await response.json();
      console.log(result);
      document.getElementById("result").textContent = result.digit;
      


      const ctx = document.getElementById("barGraph").getContext('2d');
      const data = result.arr[0];
   
      // document.getElementById("proba").textContent = result.predictions[0];


      new Chart(ctx, {
        type: 'line', 
        data: {
          labels: [0,1,2,3,4,5,6,7,8,9], 
          datasets: [{
            label: 'Prediction',
            data: data, 

            backgroundColor: "rgba(54, 162, 235, 0.2)",
            borderColor: "rgba(54, 162, 235, 1)",
            borderWidth: 1,
          }]
        },
      options: {
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
      });

    } else {
      console.error("API Error:", response.status, response.statusText);
    }
  } catch (error) {
    console.error("Network Error:", error);
  }
}

// Reset canvas
function resetCanvas() {
  context.clearRect(0, 0, 200, 200);
  clickX = [];
  clickY = [];
  clickDrag = [];
}
