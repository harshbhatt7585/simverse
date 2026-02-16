import styles from "./PageHeading.module.css";

type PageHeadingProps = {
  lineOne: string;
  lineTwo: string;
  highlightWord?: string;
};

function highlightText(text: string, target: string, className: string) {
  const index = text.toLowerCase().indexOf(target.toLowerCase());

  if (index === -1) {
    return text;
  }

  const before = text.slice(0, index);
  const match = text.slice(index, index + target.length);
  const after = text.slice(index + target.length);

  return (
    <>
      {before}
      <span className={className}>{match}</span>
      {after}
    </>
  );
}

export default function PageHeading({
  lineOne,
  lineTwo,
  highlightWord = "Environment",
}: PageHeadingProps) {
  return (
    <h1 className={styles.title}>
      <span className={styles.lineOne}>
        {highlightText(lineOne, highlightWord, styles.highlight)}
      </span>
      <span className={styles.lineTwo}>{lineTwo}</span>
    </h1>
  );
}
