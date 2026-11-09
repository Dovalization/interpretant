import { Fragment } from "react";

import { cn } from "@/lib/utils";

import type { Prediction, Predictions, Verdict } from "@/lib/schema";

const REPOSITORY_DOCS = "https://github.com/Dovalization/interpretant/blob/main/docs";

const VERDICT_STYLE: Record<Verdict, string> = {
  passed: "bg-success/15 text-success",
  falsified: "bg-destructive/15 text-destructive",
  untestable: "bg-muted text-muted-foreground",
};

/** Renders `*word*` as emphasis, the only markup the prediction titles use. */
export function Emphasised({ text }: { text: string }) {
  const parts = text.split(/(\*[^*]+\*)/g);
  return (
    <>
      {parts.map((part, index) =>
        part.startsWith("*") && part.endsWith("*") ? (
          <em key={index}>{part.slice(1, -1)}</em>
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </>
  );
}

export function VerdictBadge({ verdict, note }: { verdict?: Verdict; note?: string }) {
  if (!verdict) {
    return <span className="text-xs text-muted-foreground">not yet scored</span>;
  }
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold capitalize",
        VERDICT_STYLE[verdict],
      )}
    >
      {verdict}
      {note && <span className="font-normal normal-case opacity-80">({note})</span>}
    </span>
  );
}

export const roundTotals = (predictions: Prediction[]) => ({
  passed: predictions.filter((prediction) => prediction.verdict === "passed").length,
  falsified: predictions.filter((prediction) => prediction.verdict === "falsified").length,
  pending: predictions.filter((prediction) => !prediction.verdict).length,
});

/** Every prediction by round: id, title, verdict and the numbers it was scored on. */
export function Scorecard({ data }: { data: Predictions }) {
  return (
    <div className="flex flex-col gap-10">
      {data.rounds.map((round) => {
        const predictions = data.predictions.filter((prediction) => prediction.round === round.id);
        const totals = roundTotals(predictions);
        const file = predictions.find((prediction) => prediction.file)?.file;
        return (
          <section key={round.id}>
            <header className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
              <div>
                <h2 className="font-heading text-2xl font-semibold">{round.title}</h2>
                <p className="text-muted-foreground">{round.subtitle}</p>
              </div>
              <p className="text-sm text-muted-foreground tabular-nums">
                {totals.passed} passed · {totals.falsified} falsified
                {totals.pending > 0 && ` · ${totals.pending} pending`}
                {file && (
                  <>
                    {" · "}
                    <a
                      href={`${REPOSITORY_DOCS}/${file}`}
                      className="underline underline-offset-4 hover:text-foreground"
                    >
                      source
                    </a>
                  </>
                )}
              </p>
            </header>
            <div className="overflow-hidden rounded-lg border">
              <table className="w-full text-sm">
                <tbody>
                  {predictions.map((prediction) => (
                    <tr key={prediction.id} className="border-b align-top last:border-b-0">
                      <td className="w-12 px-3 py-2.5 font-mono text-xs text-muted-foreground">
                        {prediction.id}
                      </td>
                      <td className="px-3 py-2.5">
                        <p className="font-semibold">
                          <Emphasised text={prediction.title} />
                        </p>
                        {prediction.numbers && (
                          <p className="mt-1 text-muted-foreground">
                            <Emphasised text={prediction.numbers} />
                          </p>
                        )}
                      </td>
                      <td className="w-40 px-3 py-2.5 text-right">
                        <VerdictBadge verdict={prediction.verdict} note={prediction.note} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}
