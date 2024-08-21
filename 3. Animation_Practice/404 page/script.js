const setZeroAndBeamAppear = () => {
  document
    .getElementById("number-0-soaking")
    .removeEventListener("animationend", setZeroAndBeamAppear);

  document.getElementById("number-0-soaking").setAttribute("id", "number-0");

  document
    .getElementById("UFO-beam-disappearing")
    .setAttribute("id", "UFO-beam");

  document
    .getElementById("number-0")
    .addEventListener("animationend", setZeroAndBeamDisappear);
};

const setZeroAndBeamDisappear = () => {
  document
    .getElementById("number-0")
    .removeEventListener("animationend", setZeroAndBeamDisappear);

  document.getElementById("number-0").setAttribute("id", "number-0-soaking");

  document
    .getElementById("UFO-beam")
    .setAttribute("id", "UFO-beam-disappearing");

  document
    .getElementById("number-0-soaking")
    .addEventListener("animationend", setZeroAndBeamAppear);
};

document
  .getElementById("number-0")
  .addEventListener("animationend", setZeroAndBeamDisappear);
