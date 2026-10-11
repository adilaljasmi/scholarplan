// Crops the dashboard captures in presentation/assets/raw_*.png into the four
// slide-6 panels. Coordinates are in the 2x-resolution capture pixels.
// Run from presentation/build:  node make_demo_assets.js
const path = require("path");
const sharp = require("sharp");

const A = path.resolve(__dirname, "..", "assets");
const BG = { r: 14, g: 17, b: 23, alpha: 1 }; // Streamlit dark background (#0E1117)

async function crop(src, out, box) {
  await sharp(path.join(A, src)).extract(box).png().toFile(path.join(A, out));
  console.log(out, `${box.width}x${box.height}`);
}

// Stack several regions of one capture vertically (skipping what lies between).
async function stack(src, out, boxes, gap = 36) {
  const parts = await Promise.all(boxes.map((b) => sharp(path.join(A, src)).extract(b).png().toBuffer()));
  const width = Math.max(...boxes.map((b) => b.width));
  const height = boxes.reduce((h, b) => h + b.height, 0) + gap * (boxes.length - 1);
  let top = 0;
  const layers = boxes.map((b, i) => { const l = { input: parts[i], top, left: 0 }; top += b.height + gap; return l; });
  await sharp({ create: { width, height, channels: 4, background: BG } }).composite(layers).png().toFile(path.join(A, out));
  console.log(out, `${width}x${height}`);
}

(async () => {
  // 1. Planner Thought -> create_plan -> Thought -> issue_subtask
  await crop("raw_trace.png", "demo_1_plan.png", { left: 165, top: 1258, width: 1870, height: 645 });
  // 2. Claims list with the planted claim rejected, Critic reason and real source
  await crop("raw_claims.png", "demo_2_critic.png", { left: 148, top: 962, width: 1892, height: 616 });
  // 3. Re-planning run: metric row (Re-plans 2) + the Processor feedback the Planner read
  await stack("raw_feedback.png", "demo_3_replan.png", [
    { left: 154, top: 423, width: 1892, height: 127 },
    { left: 154, top: 935, width: 1892, height: 376 },
  ]);
  // 4. Report run line + Answer paragraph with merged Harvard citations. The
  //    big title and grey disclaimer are skipped so the text (and the claim-id
  //    superscripts) render large enough to read on the slide.
  await stack("raw_report_a.png", "demo_4_report.png", [
    { left: 150, top: 1303, width: 1898, height: 71 },
    { left: 150, top: 1560, width: 1898, height: 406 },
  ], 28);
})().catch((e) => { console.error(e); process.exit(1); });
