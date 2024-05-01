
/* The `Streamlit` object exists because our html file includes `streamlit-component-lib.js`.
// If you get an error about "Streamlit" not being defined, that means you're missing that file.

/*Send the click coordinates and plot data to Streamlit.*/
function sendClickCoordinates(clickCoords) {
  Streamlit.setComponentValue(clickCoords);
}

// Initialize SVG dimensions
const width = 300;
const height = 300;
let origin = {x: width / 2, y: height / 2};

function clearSVG(){
  // Select the SVG plot and remove all its child elements
  d3.select("#plot1").selectAll("*").remove();
  //Refresh the page
  location.reload(); 
  }

function onRender3Click(event) {

// Draw basis vectors and start with a CLICK ON ME text
const s1 = { x: origin.x + 100, y: origin.y };
const s2 = { x: origin.x, y: origin.y - 100 };

// Initialize the SVG
  const svg = d3.select("#plot1")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

// Draw "CLICK ON ME" text
svg.append("text")
  .attr("x", width/1.8)
  .attr("y", height/1.8)
  .attr("text-anchor", "middle")
  .attr("fill", "gray")
  .attr("font", "13px")
  .text("CLICK ON ME");

  drawVector(svg, origin, s1, "s1", "black");
  drawVector(svg, origin, s2, "s2", "black");

  // Add click event listener to the canvas
  svg.on("click", canvasClick);
  

  // Initialize variables
  let clickCount = 0;
  const vectors = [];

  function canvasClick(event) {
    const coords = d3.pointer(event);
    vectors.push({ x: coords[0], y: coords[1] });
    clickCount ++;

    //If length of vectors is 1, plot the first vector
    if (clickCount === 1){
    svg.text(" ")
    drawVector(svg, origin, { x: coords[0], y: coords[1] }, `v${clickCount}`, "green");
    }
  
    // Draw the two basis vectors 
    else if (clickCount === 2) {
      let vector1 = vectors[0]
      let vector2 = vectors[1]
      const v1Norm = math.norm([vector1.x, vector1.y])
      const v2Norm = math.norm([vector2.x, vector2.y])
      const cos_angle = math.dot([vector1.x, vector1.y], [vector2.x, vector2.y]) / (v1Norm * v2Norm)

          // plot the second vector iff cos value is not 1 else print an error message
          if (cos_angle === 1 ) {
              svg.text("WHOOPS! You entered parallel vectors, which cannot be orthogonalized. Please try again!")
                   }
          else if  (cos_angle == 0){
            vectors = [];
            svg.text("Wooow! You entered vectors that are already orthogonal. Please try again!")
          }
          else  {
              drawVector(svg, origin, { x: coords[0], y: coords[1] }, `v${clickCount}`, "green");
              } 
    }
    else if (clickCount === 3){
      drawVector(svg, origin, {x: coords[0], y:coords[1]}, 'x', "red")
      // Send the click coordinates and plot data to Streamlit
      sendClickCoordinates(vectors);
    }
    
      } 

  clickCount = 0
}

 //Function to draw vectors
 function drawVector(plot, start, end, id, color) {

  plot.append("line")
   .attr("x1", start.x)
   .attr("y1", start.y)
   .attr("x2", end.x)
   .attr("y2", end.y)
   .attr("stroke", color)
   .attr("stroke-width", 2)
   .attr("marker-end", `url(#arrowhead-${color})`)
   .attr("id", id);


    //Check to see if the annotation text already exists, if not, append it
   if (plot.select(`#${id}-text`).empty()) {
       plot.append("text")
       .attr("id", `${id}-text`)
       .attr("x", end.x - 12)
       .attr("y", end.y - 12)
       .text(id)
       .attr("fill", color)
       .attr("font-family", "serif")
       .attr("font-size", "15px");
   }
   
   // Create arrowhead marker for each color
  plot.append("defs").selectAll("marker")
  .data([`${color}`])
  .enter().append("marker")
  .attr("id", d => `arrowhead-${d}`)
  .attr("viewBox", "0 0 10 10")
  .attr("refX", 9)
  .attr("refY", 5)
  .attr("markerWidth", 6)
  .attr("markerHeight", 6)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M 0 0 L 10 5 L 0 10 z")
  .style("fill", `${color}`);

}


// Render the component whenever Python sends a "render event"
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender3Click);
// Tell Streamlit that the component is ready to receive events
Streamlit.setComponentReady();

  //End of mouse click

  //Start of Mouse drag events
  // function canvasMouseDown(event) {
  //   isDragging = true;
  //   const coords = d3.pointer(event);
  //   vectors.push({ x: coords[0], y: coords[1] });
  //   // Handle mouse down event
  //   console.log("Mouse drag start coordinates:", coords);
  //   // You can perform further actions here, such as updating the UI or storing the coordinates
  // }

  // function canvasMouseMove(event) {
  //   if (isDragging) {
  //       const coords = d3.pointer(event);
  //       // Handle mouse move event
  //       console.log("Mouse dragging coordinates:", coords);
  //       // You can perform further actions here, such as updating the UI or storing the coordinates
  //   }
  // }

  // function canvasMouseUp(event) {
  //   isDragging = false;
  //   const coords = d3.pointer(event);
  //   vectors.push({ x: coords[0], y: coords[1] })
  //   // Handle mouse up event
  //   console.log("Mouse drag end coordinates:", coords);
  //   // You can perform further actions here, such as updating the UI or storing the coordinates