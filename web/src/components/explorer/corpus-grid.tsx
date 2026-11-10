import { HorizontalScroll } from "@/components/horizontal-scroll";
import { decadeLabel, fieldLabel } from "@/lib/fields";
import { cn } from "@/lib/utils";

import type { Cell } from "@/lib/schema";

interface CorpusGridProps {
  cells: Cell[];
  fields: string[];
  decades: number[];
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

const number = new Intl.NumberFormat("en-US");

/** Documents per field × decade cell as a heatmap table: one blue ramp, counts in the cells. */
export function CorpusGrid({ cells, fields, decades }: CorpusGridProps) {
  const byKey = new Map(cells.map((cell) => [`${cell.field}:${cell.decade}`, cell]));
  const max = Math.max(...cells.map((cell) => cell.documents ?? 0), 1);

  return (
    <HorizontalScroll>
      <table className="w-full border-separate border-spacing-0.5 text-sm">
        <thead>
          <tr>
            <th scope="col" className="sr-only">
              Field
            </th>
            {decades.map((decade) => (
              <th
                key={decade}
                scope="col"
                className="pb-1 text-center text-xs font-semibold text-muted-foreground"
              >
                {decadeLabel(decade)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fields.map((field) => (
            <tr key={field}>
              <th
                scope="row"
                className="pr-3 text-right text-xs font-semibold whitespace-nowrap text-muted-foreground"
              >
                {fieldLabel(field)}
              </th>
              {decades.map((decade) => {
                const cell = byKey.get(`${field}:${decade}`);
                const documents = cell?.documents ?? 0;
                const step = documents === 0 ? -1 : Math.min(6, Math.floor((documents / max) * 7));
                return (
                  <td
                    key={decade}
                    title={
                      cell
                        ? `${fieldLabel(field)}, ${decadeLabel(decade)}: ${number.format(documents)} abstracts, ${number.format(cell.tokens)} tokens`
                        : `${fieldLabel(field)}, ${decadeLabel(decade)}: no cell`
                    }
                    className={cn(
                      "h-9 min-w-14 rounded-sm text-center tabular-nums",
                      step < 0 && "text-muted-foreground/60",
                      step >= 4 && "text-white",
                      step >= 0 && step < 4 && "text-neutral-950",
                    )}
                    style={step >= 0 ? { backgroundColor: RAMP[step] } : undefined}
                  >
                    {documents === 0 ? "–" : number.format(documents)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </HorizontalScroll>
  );
}
