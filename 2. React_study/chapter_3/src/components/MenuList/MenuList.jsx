import "./MenuList.css";
import { MenuListItem } from "../MenuListItm/MenuListItem";

export function MenuList(props) {
  const menus = ["좋아요!", "정말 좋아요!", "최고에요!", "미쳤어요!"];

  return (
    <ul className="container-list">
      {menus.map((moodEl) => {
        return (
          <MenuListItem
            key={moodEl}
            clickEvent={props.onItemClick}
            mood={moodEl}
          />
        );
      })}
    </ul>
  );
}
