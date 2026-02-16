import styles from "./PageHeading.module.css";

type PageHeadingProps = {
  text: string;
};

export default function PageHeading({ text }: PageHeadingProps) {
  return <h1 className={styles.title}>{text}</h1>;
}
