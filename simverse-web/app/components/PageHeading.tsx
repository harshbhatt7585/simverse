import styles from "./PageHeading.module.css";

type PageHeadingProps = {
  lineOne: string;
  lineTwo: string;
};

export default function PageHeading({ lineOne, lineTwo }: PageHeadingProps) {
  return (
    <h1 className={styles.title}>
      <span className={styles.lineOne}>{lineOne}</span>
      <span className={styles.lineTwo}>{lineTwo}</span>
    </h1>
  );
}
