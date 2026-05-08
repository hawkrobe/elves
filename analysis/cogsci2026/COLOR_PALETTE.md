# Color and line coding (Experiments 1 & 2)

**Convention:** Reserve **color** for one conceptual dimension and use **linetype** (and optionally **shape**) for others so the same visual channel doesn’t encode two things.

---

## Recommended mapping

| Dimension | Visual channel | Rationale |
|-----------|----------------|------------|
| **Frequency** (HF vs LF, 1:2 vs 1:4) | **Color** (red / blue) | Single “frequency” concept: HF/red, LF/blue; 1:2 vs 1:4 can use same hue family or red/blue if only two levels. |
| **Time pressure** (Strict 5s vs Lenient 10s) | **Color** (distinct from red/blue) or **linetype** | Avoid red/blue for 5s/10s so red/blue = frequency only. Use a different palette (e.g. amber vs magenta) or linetype (solid vs dashed). |
| **Payoff** (linear vs binary) | **Linetype** (solid vs dashed) | Works well with color for frequency; doesn’t compete with color. |

---

## Option A: Color for frequency only; linetype for time pressure and payoff

- **HF vs LF (or 1:2 vs 1:4):** Red vs blue  
  - e.g. `HF = "#C0392B"`, `LF = "#2980B9"` (or `#E94F37` / `#2E86AB` for slightly brighter).
- **Strict vs Lenient (5s vs 10s):** Linetype  
  - e.g. Strict = solid, Lenient = dashed (or dotted).
- **Linear vs Binary:** Linetype (if both time pressure and payoff appear)  
  - e.g. Linear = solid, Binary = dashed; then time pressure can be shape (circle vs triangle) or a second linetype (solid vs dotted) if you need both.

**Pros:** Red/blue is exclusively “frequency”; strict/lenient and payoff are easy to read as line style.  
**Cons:** When you have many lines (e.g. 4: 2 time pressure × 2 payoff), linetype + shape can get busy.

---

## Option B: Color for frequency; different color pair for strict/lenient; linetype for payoff

- **Frequency (HF vs LF, or 1:2 vs 1:4):** Red vs blue  
  - e.g. `#C0392B` (red), `#2980B9` (blue).
- **Strict vs Lenient:** A different pair so it never overlaps with red/blue  
  - **Amber vs magenta:** Strict = `#D4A017` or `#F59E0B`, Lenient = `#9C27B0` (purple magenta).  
  - **Slate vs green:** Strict = `#475569`, Lenient = `#059669`.  
  - **Warm gray vs cool blue-gray:** Strict = `#78716C`, Lenient = `#64748B`.
- **Payoff:** Linetype (solid = linear, dashed = binary).

**Pros:** Time pressure is still encoded by color (very salient), payoff by linetype.  
**Cons:** Two color legends; need to keep palettes clearly distinct (no red/blue for time pressure).

---

## Suggested hex codes (ready to paste)

### Frequency (red / blue) — use for HF vs LF, or 1:2 vs 1:4 when only two levels

```r
# Warm red (HF or “more HF”) and cool blue (LF or “more LF”)
colors_freq <- c("HF" = "#C0392B", "LF" = "#2980B9")
# Slightly brighter option
# colors_freq <- c("HF" = "#E94F37", "LF" = "#2E86AB")

# If you need 1:2 vs 1:4 as two distinct colors (both frequency-related):
colors_freq_ratio <- c("1:2" = "#6B4C9A", "1:4" = "#2E86AB")  # purple / blue
# Or keep red/blue: "1:2" = "#C0392B", "1:4" = "#2980B9"
```

### Time pressure (strict / lenient) — use a non–red/blue pair

```r
# Amber (strict) vs magenta (lenient)
colors_tp <- c("5" = "#D4A017", "10" = "#9C27B0")

# Or slate vs green
# colors_tp <- c("5" = "#475569", "10" = "#059669")
```

### Payoff — linetype only

```r
scale_linetype_manual(
  values = c("linear" = "solid", "binary" = "dashed"),
  name = "Payoff"
)
```

### When both time pressure and payoff appear (e.g. Exp 2)

- **Color:** Frequency (red/blue) **or** time pressure (amber/magenta), depending on the main comparison in that figure.
- **Linetype:** Payoff (solid/dashed).
- **Other:** If you need the third factor (e.g. time pressure when color = frequency), use **linetype** for one and **shape** for the other, e.g.  
  - Payoff: solid vs dashed  
  - Time pressure: circle vs triangle (for points) or solid vs dotted (for lines).

---

## Summary

- **Red/blue = frequency** (HF vs LF, or 1:2 vs 1:4) keeps one clear meaning for that pair.
- **Strict/lenient = other colors** (e.g. amber/magenta) or **linetype** avoids overloading red/blue.
- **Payoff = linetype** (solid/dashed) is consistent and works well next to color-coded frequency or time pressure.

---

## Applied in race model QMDs

**`experiment1_race_model.qmd`** and **`experiment2_race_model.qmd`** now use:

- **Strict vs Lenient (time pressure):** Amber `#D4A017` (5s) / Magenta `#9C27B0` (10s).
- **HF vs LF (frequency):** Red `#C0392B` (HF) / Blue `#2980B9` (LF) — activation curves and P(LF)/P(HF) vs time.
- **Payoff** (Exp 2 only): Encoded by facet (Linear / Binary rows or columns).

Other experiment QMDs (main stats/empirical figs) still use the previous palettes; edit those separately for consistency if desired.
