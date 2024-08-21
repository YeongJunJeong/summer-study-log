import { useState } from "react";
import "./App.css";
import { DisPlayMood } from "./components/DisPlayMood/DisPlayMood";
import { MenuList } from "./components/MenuList/MenuList";

function App() {
  const [currentMood, setCurrentMood] = useState("");

  return (
    <>
      <h1>오늘의 기분을 선택해 주세요!</h1>
      <div className="app-main">
        <MenuList mood={currentMood} onItemClick={setCurrentMood} />
        <DisPlayMood mood={currentMood} />
      </div>
    </>
  );
}
export default App;
