"use client";

import { CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";

import { ChartContainer, ChartTooltipContent } from "@/components/ui/chart";
import { decadeLabel, fieldColor, fieldLabel } from "@/lib/fields";

import type { WithinRow } from "@/lib/schema";

interface TrajectoryChartProps {
  field: string;
  rows: WithinRow[];
}

/**
 * One field's drift per transition, as percentile ranks against the pair's noise floor:
 * cosine drift (solid) and neighbourhood shift (dashed), both on one 0–100 axis. A
 * transition is a candidate when both exceed the 95th percentile (the reference line).
 */
export function TrajectoryChart({ field, rows }: TrajectoryChartProps) {
  const color = fieldColor(field);
  const data = rows.map((row) => ({
    transition: `${decadeLabel(row.decadeA)}→${row.decadeB}s`,
    cosineRank: Math.round(row.cosineRank),
    shiftRank: Math.round(row.shiftRank),
    candidate: row.candidate,
    counts: `${row.countA.toLocaleString("en-US")} → ${row.countB.toLocaleString("en-US")}`,
  }));
  const config = {
    cosineRank: { label: "cosine drift rank", color },
    shiftRank: { label: "neighbourhood shift rank", color },
  };

  if (rows.length === 0) {
    return (
      <p className="py-6 text-sm text-muted-foreground">
        {fieldLabel(field)}: no transition with 150 occurrences on both sides.
      </p>
    );
  }

  return (
    <div>
      <p className="mb-1 text-sm font-semibold" style={{ color }}>
        {fieldLabel(field)}
      </p>
      <ChartContainer config={config} className="aspect-[2/1] w-full">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
          <CartesianGrid vertical={false} strokeDasharray="0" />
          <XAxis dataKey="transition" tickLine={false} axisLine={false} fontSize={11} />
          <YAxis domain={[0, 100]} ticks={[0, 50, 95]} tickLine={false} axisLine={false} />
          <ReferenceLine y={95} stroke="var(--color-muted-foreground)" strokeDasharray="4 4" />
          <Tooltip content={<ChartTooltipContent />} cursor={{ stroke: "var(--color-border)" }} />
          <Line
            type="monotone"
            dataKey="cosineRank"
            stroke={color}
            strokeWidth={2}
            dot={(props: { cx?: number; cy?: number; payload?: { candidate: boolean } }) => (
              <circle
                key={`${props.cx}-${props.cy}`}
                cx={props.cx}
                cy={props.cy}
                r={props.payload?.candidate ? 6 : 3}
                fill={color}
                stroke="var(--color-background)"
                strokeWidth={2}
              />
            )}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="shiftRank"
            stroke={color}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}
