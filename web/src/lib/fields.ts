/**
 * Field identity: label and colour. A field keeps its colour in every view (the chart
 * palette is never cycled); fields past the eighth slot share a neutral and are told apart
 * by their labels, which every chart shows.
 */

export const FIELD_LABELS: Record<string, string> = {
  ai: "AI",
  books: "Books",
  cogpsych: "Cognitive psychology",
  communication: "Communication",
  compsci: "Computer science",
  hci: "HCI",
  histphilsci: "History & philosophy of science",
  linguistics: "Linguistics",
  neuroscience: "Neuroscience",
  philosophy: "Philosophy",
  psychology: "Psychology",
  sociology: "Sociology",
};

const FIELD_SLOTS: Record<string, number> = {
  philosophy: 1,
  hci: 2,
  ai: 3,
  compsci: 4,
  psychology: 5,
  cogpsych: 6,
  neuroscience: 7,
  linguistics: 8,
};

export const fieldLabel = (field: string): string => FIELD_LABELS[field] ?? field;

/**
 * CSS colour for a field: a chart slot, or the muted foreground for unslotted fields. The
 * raw `--chart-N` tokens are used (not Tailwind's `--color-chart-N`), because Tailwind 4 only
 * emits theme variables that a utility class references.
 */
export const fieldColor = (field: string): string => {
  const slot = FIELD_SLOTS[field];
  return slot === undefined ? "var(--muted-foreground)" : `var(--chart-${slot})`;
};

export const decadeLabel = (decade: number): string => `${decade}s`;

export const LAYER_LABELS = {
  openalex: "Main grid",
  openalex_full: "Full cells",
  books: "Books",
} as const;

export const METHOD_LABELS = {
  twec: "TWEC",
  ppmi: "PPMI-SVD",
} as const;
