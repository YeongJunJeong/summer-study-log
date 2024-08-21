export default function Resume(props) {
  return (
    <div>
      <h1>{props.name}의 자기소개서</h1>
      <h2>{props.hello}</h2>
      <dl>
        <dt>취미 :</dt>
        <dd>{props.hobby}</dd>
        <dt>좋아하는 음식 :</dt>
        <dd>{props.food}</dd>
        <dt>좋아하는 색 :</dt>
        <dd style={{ color: props.color }}>{props.color}</dd>
      </dl>
    </div>
  );
}
