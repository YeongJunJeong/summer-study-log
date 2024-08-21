export default function HelloProps(props) {
  return (
    <div>
      <p>
        {props.name} {props.age}
      </p>
      <strong>{props.someFunc()}</strong>
      <p>{[...props.someArr]}</p>
      <p>{props.someObj.one}</p>
    </div>
  );
}
