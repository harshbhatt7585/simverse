import PageHeading from "./components/PageHeading";
import PromptInput from "./components/PromptInput";
import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.main}>
      <PageHeading text="Build RL environments with Simverse" />
      <PromptInput placeholder="Describe your environment..." />
    </main>
  );
}
