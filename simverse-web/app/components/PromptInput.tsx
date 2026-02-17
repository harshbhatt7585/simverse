import styles from "./PromptInput.module.css";

type PromptInputProps = {
  placeholder: string;
};

export default function PromptInput({ placeholder }: PromptInputProps) {
  return (
    <section className={styles.composer} aria-label="Environment prompt composer">
      <textarea
        id="environment-prompt"
        className={styles.textarea}
        placeholder={placeholder}
        aria-label="Environment description"
        rows={6}
      />
    </section>
  );
}
