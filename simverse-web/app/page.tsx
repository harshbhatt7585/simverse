import PageHeading from "./components/PageHeading";
import PromptInput from "./components/PromptInput";
import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.page}>
      <video
        className={styles.videoBg}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
      >
        <source src="/videos/48354-453189085.mp4" type="video/mp4" />
      </video>
      <div className={styles.overlay} />

      <main className={styles.main}>
        <PageHeading text="Build RL Environments using Natural Language and train in one Click" />
        <PromptInput placeholder="Describe your environment..." />
      </main>
    </div>
  );
}
