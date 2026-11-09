import type { ReactNode } from "react";

interface Stat {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}

/** A row of headline numbers in bordered cells (the arch-type stat grid). */
export function StatStrip({ stats }: { stats: Stat[] }) {
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-lg border bg-card sm:grid-cols-4">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="flex flex-col gap-1 border-r border-b p-4 last:border-r-0 sm:border-b-0 nth-[2n]:border-r-0 sm:nth-[2n]:border-r"
        >
          <dt className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {stat.label}
          </dt>
          <dd className="font-heading text-3xl font-semibold text-foreground">{stat.value}</dd>
          {stat.detail && <dd className="text-sm text-muted-foreground">{stat.detail}</dd>}
        </div>
      ))}
    </dl>
  );
}
