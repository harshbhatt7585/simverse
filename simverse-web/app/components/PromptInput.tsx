import styles from "./PromptInput.module.css";

type PromptInputProps = {
  placeholder: string;
};

export default function PromptInput({ placeholder }: PromptInputProps) {
  return (
    <input
      className={styles.input}
      type="text"
      placeholder={placeholder}
      aria-label="Environment description"
    />
  );
}
