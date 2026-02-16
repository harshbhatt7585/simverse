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
        <source
          src="https://cdn.coverr.co/videos/coverr-clouds-and-sunset-over-mountains-1579/1080p.mp4"
          type="video/mp4"
        />
      </video>
      <div className={styles.overlay} />

      <main className={styles.main}>
        <PageHeading text="Build RL environments with Simverse" />
        <PromptInput placeholder="Describe your environment..." />
      </main>
    </div>
  );
}
