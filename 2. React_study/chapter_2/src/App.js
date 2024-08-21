import Hello from "./components/Hello";
import HelloProps from "./components/HelloProps";
import Time from "./components/Time";
import Resume from "./components/Resume";
import { ColorText } from "./components/ColorText";

function App() {
  return (
    <div>
      <HelloProps
        name="yeongjun"
        age={24}
        someFunc={() => "hello!!"}
        someJSX={<img src="https://picsum.photos/id/237/200/300" />}
        someArr={[1, 2, 3]}
        someObj={{ one: 1 }}
      />
      <Hello name="gary" />
      <Time />
      <Resume
        hello="안녕하세요"
        name="개리"
        hobby="게임"
        food="고기"
        color="blue"
      />
      <ColorText color="skyblue" />
      <ColorText color="red" />
      <ColorText color="green" />
    </div>
  );
}

export default App;
