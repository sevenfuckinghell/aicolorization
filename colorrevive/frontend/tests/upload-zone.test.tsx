import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UploadZone } from "@/components/upload-zone";

describe("UploadZone", () => {
  it("renders the drop zone, browse button and format hints", () => {
    render(
      <UploadZone
        file={null}
        onFileSelected={vi.fn()}
        onRemove={vi.fn()}
        onError={vi.fn()}
      />,
    );
    expect(
      screen.getByText(/Drag & drop a black-and-white photo here/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /browse files/i })).toBeInTheDocument();
    expect(screen.getByText(/JPG, JPEG, PNG, WEBP/i)).toBeInTheDocument();
    expect(screen.getByText(/Maximum file size: 10 MB/i)).toBeInTheDocument();
  });

  it("accepts a valid PNG file via the hidden input", async () => {
    const onFileSelected = vi.fn();
    const onError = vi.fn();
    render(
      <UploadZone
        file={null}
        onFileSelected={onFileSelected}
        onRemove={vi.fn()}
        onError={onError}
      />,
    );
    const file = new File(["dummy"], "photo.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, file);
    expect(onFileSelected).toHaveBeenCalledTimes(1);
    expect(onFileSelected.mock.calls[0][0].name).toBe("photo.png");
    expect(onError).not.toHaveBeenCalled();
  });

  it("rejects unsupported file types with an error callback", async () => {
    const onFileSelected = vi.fn();
    const onError = vi.fn();
    render(
      <UploadZone
        file={null}
        onFileSelected={onFileSelected}
        onRemove={vi.fn()}
        onError={onError}
      />,
    );
    const file = new File(["<svg/>"], "drawing.svg", { type: "image/svg+xml" });
    Object.defineProperty(file, "size", { value: 100 });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, "files", { value: [file] });
    fireEvent.change(input);
    expect(onFileSelected).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(
      expect.stringContaining("Unsupported file type"),
    );
  });

  it("shows the loaded-file state and can remove it", () => {
    const onRemove = vi.fn();
    render(
      <UploadZone
        file={{
          name: "grandma.png",
          sizeBytes: 2048,
          type: "image/png",
          width: 640,
          height: 480,
          dataUrl: "data:image/png;base64,",
        }}
        onFileSelected={vi.fn()}
        onRemove={onRemove}
        onError={vi.fn()}
      />,
    );
    expect(screen.getByText(/grandma\.png/)).toBeInTheDocument();
    expect(screen.getByText(/640 × 480 px/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /remove image/i }));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });
});
