// The `Streamlit` object exists because our html file includes
// `streamlit-component-lib.js`.
// If you get an error about "Streamlit" not being defined, that means you're missing that file.

/* Send the click coordinates and plot data to Streamlit */
function sendClickCoordinates(clickCoords) {
  Streamlit.setComponentValue(clickCoords);
}

/** The click event listener for the canvas. **/
const width = int;
const height = int;
const origin = { x: width / 2, y: height / 2 };

// Initialize clickCount and vectors for 2 clicks
let clickCount3 = 0;
let vectors3 = [];

function onRender3Click(event) {
    //clearSVG("#plot2"); 

    // Initialize the SVG
    const svg = d3.select("#plot2")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

    // Draw "CLICK ON ME" text
    svg.append("text")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .attr("fill", "gray")
    .text("CLICK ON ME");

    // Add click event listener to the canvas
    svg.on("click", canvasClick3Click);

    function canvasClick3Click(event) {
      const coords = d3.pointer(event);
      vectors3.push({ x: coords[0], y: coords[1] });
      clickCount3++;

      // If length of vectors is 1, plot the first vector
      if (clickCount3 === 1) {
        svg.text(" ")
        drawVector(svg, { x: width / 2, y: height / 2 }, { x: coords[0], y: coords[1] }, `y${clickCount3}`, "green");
      }

      // Draw the two basis vectors 
      else if (clickCount3 === 2) {
        text.text("ONCE MORE")

        const vector1 = vectors3[0]
        const vector2 = vectors3[1]
        const v1Norm = math.norm([vector1.x, vector1.y])
        const v2Norm = math.norm([vector2.x, vector2.y])
        const cos_angle = math.dot([vector1.x, vector1.y], [vector2.x, vector2.y]) / (v1Norm * v2Norm)

          // plot the second vector iff cos value is not 1 else print an error message
          if (cos_angle === 1 ) {
              const vectors = [];
              svg.text("WHOOPS! You entered parallel vectors, which cannot form a basis. Please try again!")
                      }
          else  {
                drawVector(svg, origin, { x: coords[0], y: coords[1] }, `v${clickCount3}`, "green");
                } 
                            } 
    // Third click draw the vector to be expanded
    else if (clickCount3 === 3) {
      text.text(" ")
      drawVector(svg, origin, { x: coords[0], y: coords[1] }, `x${clickCount3}`, "red");
      sendClickCoordinates(vectors3);
    }
  }
}


Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender3Click);
// Tell Streamlit that the component is ready to receive events
Streamlit.setComponentReady();
