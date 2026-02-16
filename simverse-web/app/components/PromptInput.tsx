import styles from "./PromptInput.module.css";

type PromptInputProps = {
  placeholder: string;
};

export default function PromptInput({ placeholder }: PromptInputProps) {
  return (
    <textarea
      className={styles.textarea}
      placeholder={placeholder}
      aria-label="Environment description"
      rows={5}
    />
  );
}
