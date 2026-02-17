import Link from "next/link";
import styles from "./SimverseLogo.module.css";

export default function SimverseLogo() {
  return (
    <Link className={styles.logo} href="/" aria-label="Simverse home">
      <span className={styles.name}>Simverse</span>
    </Link>
  );
}
