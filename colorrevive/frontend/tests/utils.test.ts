import { describe, expect, it } from "vitest";
import {
  base64ToBlob,
  buildDownloadFilename,
  clamp,
  cn,
  formatBytes,
  validateImageFile,
} from "@/lib/utils";

function makeFile(
  name: string,
  type: string,
  sizeBytes = 1024,
): File {
  return new File([new Uint8Array(Math.min(sizeBytes, 1024))], name, {
    type,
    ...(sizeBytes > 1024 ? {} : {}),
  });
}

describe("formatBytes", () => {
  it("formats bytes, KB and MB", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("validateImageFile", () => {
  it("accepts png/jpg/webp with matching extension", () => {
    expect(validateImageFile(makeFile("a.png", "image/png")).ok).toBe(true);
    expect(validateImageFile(makeFile("b.jpg", "image/jpeg")).ok).toBe(true);
    expect(validateImageFile(makeFile("c.webp", "image/webp")).ok).toBe(true);
  });

  it("rejects unsupported types (svg, gif)", () => {
    const svg = validateImageFile(makeFile("x.svg", "image/svg+xml"));
    expect(svg.ok).toBe(false);
    expect(svg.errorCode).toBe("UNSUPPORTED_TYPE");
    const gif = validateImageFile(makeFile("x.gif", "image/gif"));
    expect(gif.ok).toBe(false);
  });

  it("rejects empty files", () => {
    const empty = validateImageFile(makeFile("e.png", "image/png", 0));
    expect(empty.ok).toBe(false);
    expect(empty.errorCode).toBe("EMPTY_FILE");
  });

  it("rejects oversized files with a human-readable message", () => {
    const big = makeFile("big.png", "image/png");
    Object.defineProperty(big, "size", { value: 11 * 1024 * 1024 });
    const res = validateImageFile(big, 10);
    expect(res.ok).toBe(false);
    expect(res.errorCode).toBe("TOO_LARGE");
    expect(res.message).toContain("10 MB");
  });

  it("rejects spoofed extensions (mime/extension mismatch)", () => {
    const res = validateImageFile(makeFile("evil.png", "application/x-msdownload"));
    expect(res.ok).toBe(false);
  });
});

describe("buildDownloadFilename", () => {
  it("produces colorrevive-<stem>-colorized.<ext>", () => {
    expect(buildDownloadFilename("grandma 1952.JPG", "png")).toBe(
      "colorrevive-grandma-1952-colorized.png",
    );
    expect(buildDownloadFilename("photo.webp", "jpeg")).toBe(
      "colorrevive-photo-colorized.jpg",
    );
  });
});

describe("base64ToBlob", () => {
  it("decodes a base64 payload into typed bytes", () => {
    const blob = base64ToBlob(btoa("hello"), "text/plain");
    expect(blob.size).toBe(5);
  });
});

describe("cn / clamp", () => {
  it("joins truthy classes", () => {
    expect(cn("a", false, undefined, "b")).toBe("a b");
  });
  it("clamps numbers", () => {
    expect(clamp(-5, 0, 100)).toBe(0);
    expect(clamp(150, 0, 100)).toBe(100);
  });
});
