import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SettingsPanel } from "@/components/settings-panel";
import { ProcessingState } from "@/components/processing-state";
import { ResultActions } from "@/components/result-actions";
import { DEFAULT_SETTINGS, type ColorizeResult } from "@/lib/types";

describe("SettingsPanel", () => {
  function setup(overrides = {}) {
    const onChange = vi.fn();
    const onStart = vi.fn();
    render(
      <SettingsPanel
        settings={{ ...DEFAULT_SETTINGS, ...overrides }}
        onChange={onChange}
        onStart={onStart}
        processing={false}
        canSubmit={true}
      />,
    );
    return { onChange, onStart };
  }

  it("renders quality options with Standard selected by default", () => {
    setup();
    const standard = screen.getByRole("radio", { name: /standard/i });
    const high = screen.getByRole("radio", { name: /high/i });
    expect(standard).toBeChecked();
    expect(high).not.toBeChecked();
  });

  it("toggles quality and preserves-contrast settings", async () => {
    const { onChange } = setup();
    await userEvent.click(screen.getByRole("radio", { name: /high/i }));
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ quality: "high" }),
    );
    const contrast = screen.getByLabelText(/Preserve original contrast/i);
    expect(contrast).toBeChecked();
    await userEvent.click(contrast);
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ preserveContrast: false }),
    );
  });

  it("starts colorization when the button is clicked", async () => {
    const { onStart } = setup();
    await userEvent.click(
      screen.getByRole("button", { name: /colorize/i }),
    );
    expect(onStart).toHaveBeenCalledTimes(1);
  });

  it("disables submission while processing or without a file", () => {
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={vi.fn()}
        onStart={vi.fn()}
        processing={false}
        canSubmit={false}
      />,
    );
    expect(
      screen.getByRole("button", { name: /colorize/i }),
    ).toBeDisabled();
  });
});

describe("ProcessingState", () => {
  it("is hidden when idle with no error", () => {
    const { container } = render(
      <ProcessingState
        phase="idle"
        uploadProgress={null}
        errorMessage={null}
        onDismissError={vi.fn()}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("announces the current phase in an ARIA live region", () => {
    render(
      <ProcessingState
        phase="running-model"
        uploadProgress={null}
        errorMessage={null}
        onDismissError={vi.fn()}
      />,
    );
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(within(status).getByText(/Running colorization model/i)).toBeInTheDocument();
    expect(within(status).getByText(/Uploading image/i)).toBeInTheDocument();
  });

  it("shows a dismissible error alert", () => {
    const onDismiss = vi.fn();
    render(
      <ProcessingState
        phase="idle"
        uploadProgress={null}
        errorMessage="The backend could not be reached."
        onDismissError={onDismiss}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The backend could not be reached.",
    );
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });
});

describe("ResultActions", () => {
  const result: ColorizeResult = {
    blobUrl: "blob:mock",
    base64: "",
    mimeType: "image/png",
    filename: "colorrevive-sample-colorized.png",
    width: 640,
    height: 480,
    processingTimeMs: 281,
    model: "fallback-heuristic",
    fallbackMode: true,
    requestId: "req-1",
  };

  it("renders download and reset buttons", () => {
    fetchMock(result.blobUrl);
    render(
      <ResultActions result={result} originalName="sample.png" onReset={vi.fn()} />,
    );
    expect(screen.getByRole("button", { name: /download png/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download jpg/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start over/i })).toBeInTheDocument();
  });

  it("calls onReset when Start over is clicked", async () => {
    fetchMock(result.blobUrl);
    const onReset = vi.fn();
    render(
      <ResultActions result={result} originalName="sample.png" onReset={onReset} />,
    );
    await userEvent.click(screen.getByRole("button", { name: /start over/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});

function fetchMock(url: string) {
  // jsdom lacks URL.createObjectURL-based fetch; stub fetch for blob URLs.
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      if (String(input) === url) {
        return new Response(new Blob([new Uint8Array(8)], { type: "image/png" }));
      }
      return new Response("not found", { status: 404 });
    }),
  );
  if (typeof (globalThis as any).createImageBitmap !== "function") {
    (globalThis as any).createImageBitmap = async () => ({
      width: 8,
      height: 8,
      close() {},
    });
  }
}
