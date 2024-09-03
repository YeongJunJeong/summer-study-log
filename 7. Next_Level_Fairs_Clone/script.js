document.addEventListener("mousemove", (e) => {
  const cursorDefaultInner = document.querySelector(".cursor__default__inner");
  const cursorTraceInner = document.querySelector(".cursor__trace__inner");

  cursorDefaultInner.style.top = e.clientY + "px";
  cursorDefaultInner.style.left = e.clientX + "px";

  cursorTraceInner.style.top = e.clientY + "px";
  cursorTraceInner.style.left = e.clientX + "px";
});

const cursor = document.querySelector(".cursor");

document.addEventListener("mousedown", () => {
  cursor.classList.add("cursor--active");
});

document.addEventListener("mouseup", () => {
  cursor.classList.remove("cursor--active");
});

function createRipple(e) {
  let ripple = document.createElement("span");

  ripple.classList.add("ripple");
  cursor.appendChild(ripple);

  ripple.style.top = e.clientY - ripple.clientHeight / 2 + "px";
  ripple.style.left = e.clientX - ripple.clientWidth / 2 + "px";

  ripple.addEventListener("animationend", () => {
    cursor.removeChild(ripple);
  });
}

document.addEventListener("click", (e) => {
  createRipple(e);
});
