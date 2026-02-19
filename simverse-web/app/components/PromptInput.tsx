"use client";

import { FormEvent, useState } from "react";
import { buildAgenticTemplate } from "@/lib/agenticTemplate";
import styles from "./PromptInput.module.css";

type PromptInputProps = {
  placeholder: string;
};

export default function PromptInput({ placeholder }: PromptInputProps) {
  const [prompt, setPrompt] = useState("");
  const [generatedTemplate, setGeneratedTemplate] = useState("");
  const isSubmitDisabled = prompt.trim().length === 0;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setGeneratedTemplate(
      buildAgenticTemplate({
        environmentDescription: prompt,
      }),
    );
  };

  return (
    <section className={styles.composer} aria-label="Environment prompt composer">
      <form className={styles.form} onSubmit={handleSubmit}>
        <textarea
          id="environment-prompt"
          className={styles.textarea}
          placeholder={placeholder}
          aria-label="Environment description"
          rows={6}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
        />
        <button type="submit" className={styles.submitButton} disabled={isSubmitDisabled}>
          Submit
        </button>
      </form>

      {generatedTemplate && (
        <section className={styles.templatePanel} aria-live="polite" aria-label="Agentic template">
          <p className={styles.templateHeading}>Agentic Template</p>
          <pre className={styles.templateOutput}>{generatedTemplate}</pre>
        </section>
      )}
    </section>
  );
}
