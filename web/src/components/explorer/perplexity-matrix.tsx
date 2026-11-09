"use client";

import { fieldLabel } from "@/lib/fields";
import { perplexityMean } from "@/lib/perplexity";
import { cn } from "@/lib/utils";

import type { PerplexityMatrix as Matrix } from "@/lib/perplexity";

interface PerplexityMatrixProps {
  matrix: Matrix;
  selected: { model: string; text: string };
  onSelect: (pair: { model: string; text: string }) => void;
}

const RAMP = [
  "var(--ramp-1)",
  "var(--ramp-2)",
  "var(--ramp-3)",
  "var(--ramp-4)",
  "var(--ramp-5)",
  "var(--ramp-6)",
  "var(--ramp-7)",
];

/**
 * Rows are the field whose model does the predicting, columns the field whose held-out
 * abstracts are predicted. A cell is the excess cross-entropy in bits per token (mean of five
 * seeds): low means the column's language is predictable from the row's. Click a cell to plot
 * that pair over the decades.
 */
export function PerplexityMatrix({ matrix, selected, onSelect }: PerplexityMatrixProps) {
  const { fields, cell, max } = matrix;
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr>
            <th
              scope="col"
              className="pr-2 pb-1 text-right align-bottom font-normal text-muted-foreground"
            >
              model ↓ &nbsp; text →
            </th>
            {fields.map((field) => (
              <th
                key={field}
                scope="col"
                className="h-24 align-bottom font-semibold text-muted-foreground"
              >
                <span className="inline-block w-9 origin-bottom-left -rotate-45 whitespace-nowrap">
                  {fieldLabel(field)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((model) => (
            <tr key={model}>
              <th
                scope="row"
                className="pr-2 text-right font-semibold whitespace-nowrap text-muted-foreground"
              >
                {fieldLabel(model)}
              </th>
              {fields.map((text) => {
                const row = cell(model, text);
                const value = row ? perplexityMean(row.excess) : undefined;
                const step = value === undefined ? -1 : Math.min(6, Math.floor((value / max) * 7));
                const isSelected = selected.model === model && selected.text === text;
                return (
                  <td key={text} className="p-0">
                    {row && value !== undefined ? (
                      <button
                        type="button"
                        onClick={() => onSelect({ model, text })}
                        aria-pressed={isSelected}
                        title={`${fieldLabel(model)}'s model on ${fieldLabel(text)}'s abstracts: ${value.toFixed(2)} bits per token above ${fieldLabel(text)}'s own model`}
                        className={cn(
                          "size-9 rounded-sm text-center tabular-nums transition-[outline] focus-visible:outline-2 focus-visible:outline-ring",
                          step >= 4 ? "text-white" : "text-neutral-950",
                          isSelected && "ring-2 ring-foreground ring-inset",
                        )}
                        style={{ backgroundColor: RAMP[step] }}
                      >
                        {value.toFixed(1)}
                      </button>
                    ) : (
                      <div className="size-9 rounded-sm bg-muted/40" aria-hidden />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
