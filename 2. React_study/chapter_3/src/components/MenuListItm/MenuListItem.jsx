import "./MenuListItem.css";

export function MenuListItem(props) {
  const onItemClick = () => {
    props.clickEvent(props.mood);
  };
  return (
    <li>
      <button className="btn-item" onClick={onItemClick}>
        기분이 : {props.mood}
      </button>
    </li>
  );
}
