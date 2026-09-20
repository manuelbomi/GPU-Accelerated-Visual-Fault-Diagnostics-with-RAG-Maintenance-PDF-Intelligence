import type { RetrievedPassage } from "../types";
import "./PassagesList.css";

export function PassagesList({ passages }: { passages: RetrievedPassage[] }) {
  if (passages.length === 0) {
    return <p className="muted">No relevant maintenance passages found.</p>;
  }
  return (
    <ul className="passages-list">
      {passages.map((p, i) => (
        <li key={i}>
          <p>{p.text}</p>
          <span className="passage-score">relevance {(p.score * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}
