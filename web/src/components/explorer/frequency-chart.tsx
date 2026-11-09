"use client";

import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { decadeLabel, fieldColor, fieldLabel } from "@/lib/fields";

import type { FrequencyPoint } from "@/lib/explorer";

interface FrequencyChartProps {
  points: FrequencyPoint[];
  fields: string[];
}

/** Occurrences per million tokens per decade, one line per field. */
export function FrequencyChart({ points, fields }: FrequencyChartProps) {
  const data = points.map((point) => ({ decade: decadeLabel(point.decade), ...point.perMillion }));
  const config = Object.fromEntries(
    fields.map((field) => [field, { label: fieldLabel(field), color: fieldColor(field) }]),
  );

  return (
    <ChartContainer config={config} className="aspect-[5/2] w-full">
      <LineChart data={data} margin={{ top: 16, right: 56, bottom: 0, left: -8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="decade" tickLine={false} axisLine={false} fontSize={11} />
        <YAxis tickLine={false} axisLine={false} width={48} />
        <Tooltip content={<ChartTooltipContent />} cursor={{ stroke: "var(--color-border)" }} />
        {fields.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {fields.map((field) => (
          <Line
            key={field}
            type="linear"
            dataKey={field}
            name={fieldLabel(field)}
            stroke={fieldColor(field)}
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 0, fill: fieldColor(field) }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartContainer>
  );
}
