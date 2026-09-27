import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ComparisonSlider } from "@/components/comparison-slider";

function renderSlider() {
  return render(
    <ComparisonSlider
      beforeSrc="data:image/png;base64,aa"
      afterSrc="data:image/png;base64,bb"
      beforeAlt="Original grayscale image"
      afterAlt="Colorized image"
    />,
  );
}

describe("ComparisonSlider", () => {
  it("exposes an accessible slider handle", () => {
    renderSlider();
    const slider = screen.getByRole("slider", {
      name: /comparison position between original and colorized image/i,
    });
    expect(slider).toHaveAttribute("aria-valuemin", "0");
    expect(slider).toHaveAttribute("aria-valuemax", "100");
    expect(slider).toHaveAttribute("aria-valuenow", "50");
    expect(slider).toHaveAttribute("tabindex", "0");
  });

  it("moves with keyboard arrow keys", () => {
    renderSlider();
    const slider = screen.getByRole("slider");
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(52);
    fireEvent.keyDown(slider, { key: "ArrowLeft" });
    fireEvent.keyDown(slider, { key: "ArrowLeft" });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(48);
  });

  it("supports Shift+arrows for larger steps and Home/End", () => {
    renderSlider();
    const slider = screen.getByRole("slider");
    fireEvent.keyDown(slider, { key: "ArrowRight", shiftKey: true });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(60);
    fireEvent.keyDown(slider, { key: "End" });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(100);
    fireEvent.keyDown(slider, { key: "Home" });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(0);
  });

  it("clamps at the boundaries", () => {
    renderSlider();
    const slider = screen.getByRole("slider");
    fireEvent.keyDown(slider, { key: "Home" });
    fireEvent.keyDown(slider, { key: "ArrowLeft" });
    expect(Number(slider.getAttribute("aria-valuenow"))).toBe(0);
  });

  it("renders both images with meaningful alt text", () => {
    renderSlider();
    expect(screen.getByAltText("Original grayscale image")).toBeInTheDocument();
    expect(screen.getByAltText("Colorized image")).toBeInTheDocument();
  });
});
