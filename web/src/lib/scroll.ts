/** Edge state of a horizontal scroll strip: whether content hides past each side. */
export interface ScrollEdges {
  hiddenStart: boolean;
  hiddenEnd: boolean;
}

export const scrollEdges = (left: number, visible: number, total: number): ScrollEdges => ({
  hiddenStart: left > 1,
  hiddenEnd: left + visible < total - 1,
});

const FADE = "3rem";

/**
 * A CSS mask that fades the strip out on each side where more content hides, so a reader
 * sees there is more to scroll; no mask when everything fits.
 */
export const scrollMask = ({ hiddenStart, hiddenEnd }: ScrollEdges): string | undefined =>
  hiddenStart || hiddenEnd
    ? `linear-gradient(to right, ${hiddenStart ? "transparent" : "black"}, black ${FADE}, black calc(100% - ${FADE}), ${hiddenEnd ? "transparent" : "black"})`
    : undefined;
