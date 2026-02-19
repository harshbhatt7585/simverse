type AgenticTemplateInput = {
  environmentDescription: string;
};

function normalizeDescription(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

export function buildAgenticTemplate({ environmentDescription }: AgenticTemplateInput): string {
  const normalizedDescription = normalizeDescription(environmentDescription);
  const scenario = normalizedDescription || "[ADD ENVIRONMENT DETAILS]";

  return [
    "SYSTEM ROLE: RL Environment Builder Agent",
    "",
    "GOAL",
    "Convert the user request into a build-ready RL environment spec.",
    "",
    "USER REQUEST",
    scenario,
    "",
    "OUTPUT FORMAT",
    "1. Environment Name",
    "2. Objective",
    "3. Agent Action Space",
    "4. Observation Space",
    "5. Reward Function",
    "6. Episode Start/Termination Conditions",
    "7. Difficulty Modes",
    "8. Baseline Policy",
    "9. Evaluation Metrics",
    "10. First Implementation Tasks",
    "",
    "CONSTRAINTS",
    "- Keep behavior deterministic unless stochasticity is explicitly required.",
    "- Prefer sparse + shaped reward decomposition with clear coefficients.",
    "- Define edge cases and invalid actions explicitly.",
    "- Include at least one sanity-check baseline.",
  ].join("\n");
}
