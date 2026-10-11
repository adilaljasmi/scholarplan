// Builds presentation/ScholarPlan_Unit11_Presentation.pptx
// Run from presentation/build:  node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const SHOTS = path.join(ROOT, "evidence", "screenshots");
const OUT = path.join(ROOT, "presentation", "ScholarPlan_Unit11_Presentation.pptx");

const THEME = {
  name: "ScholarPlan Ink",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "334155", lt2: "F1F5F9",
    accent1: "0E7C7B", accent2: "F2A541", accent3: "2563EB", accent4: "16A34A",
    accent5: "9333EA", accent6: "EA580C", hlink: "0E7C7B", folHlink: "6D28D9",
  },
};
// Agent identity colours match the dashboard (deliberately hex: identity, not theme).
const AG = {
  planner: { hex: "2563EB", name: "Planner", icon: fa.FaCompass },
  retriever: { hex: "16A34A", name: "Retriever", icon: fa.FaMagnifyingGlass },
  processor: { hex: "9333EA", name: "Processor", icon: fa.FaGear },
  critic: { hex: "EA580C", name: "Critic", icon: fa.FaShieldHalved },
  archivist: { hex: "DC2626", name: "Archivist", icon: fa.FaDatabase },
};
const MUTED = "64748B";

async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ------------------------------------------------------------- notes
function notesBySlide() {
  const t = fs.readFileSync(path.join(ROOT, "presentation", "TRANSCRIPT.md"), "utf8").split("### References")[0];
  const parts = t.split(/\n## Slide (\d+): /).slice(1);
  const out = {};
  for (let i = 0; i < parts.length; i += 2) {
    const body = parts[i + 1].split("\n").slice(1).join("\n")
      .replace(/^---\s*$/gm, "").replace(/\*\[/g, "[").replace(/\]\*/g, "]").replace(/\*/g, "").trim();
    out[Number(parts[i])] = body;
  }
  return out;
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.author = "Adil Aljasmi";
  pres.title = "ScholarPlan: from design to a working multi-agent system";
  pres.subject = "Intelligent Agents, Unit 11 individual presentation";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const NOTES = notesBySlide();

  const icons = {};
  for (const [k, a] of Object.entries(AG)) {
    icons[k + "_w"] = await icon(a.icon, "FFFFFF");
  }
  const I = async (comp, hex) => icon(comp, hex);

  // ----------------------------------------------------------- layouts
  const FOOT = "ScholarPlan · Intelligent Agents · Unit 11 · Adil Aljasmi";
  pres.defineSlideMaster({
    title: "Title Dark",
    background: { color: THEME.colors.dk1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.9, w: 7.2, h: 1.4, fontSize: 54, bold: true, color: C.background1, fontFace: "Cambria", valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.45, w: 7.0, h: 1.2, fontSize: 22, color: C.accent2, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "Content Light",
    background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text1, fontFace: "Cambria", valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
  });
  pres.defineSlideMaster({
    title: "Content Dark",
    background: { color: THEME.colors.dk1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.background1, fontFace: "Cambria", valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: "94A3B8", margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "94A3B8", align: "right" },
  });

  const agentDot = (slide, key, x, y, d) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: AG[key].hex }, line: { color: AG[key].hex }, objectName: `${key} dot` });
    const p = d * 0.26;
    slide.addImage({ data: icons[key + "_w"], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: `${key} icon` });
  };
  const iconCircle = async (slide, comp, fillHex, x, y, d) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fillHex }, line: { color: fillHex } });
    const p = d * 0.25;
    slide.addImage({ data: await I(comp, "FFFFFF"), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
  };
  const cite = (slide, text, dark = false) =>
    slide.addText(text, { x: 0.6, y: 6.68, w: 12.1, h: 0.3, fontSize: 10, italic: true, color: dark ? "94A3B8" : MUTED, margin: 0, isTextBox: true });

  // ================================================================ 1
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Introduction" });
  s.addText("ScholarPlan", { placeholder: "title" });
  s.addText("From design to a working LLM-powered multi-agent system for academic research", { placeholder: "body" });
  s.addText([
    { text: "Intelligent Agents · Unit 11 individual presentation", options: { breakLine: true } },
    { text: "Adil Aljasmi · October 2026" },
  ], { x: 0.7, y: 5.3, w: 7, h: 0.8, fontSize: 16, color: "CBD5E1", margin: 0, isTextBox: true });
  // motif: five agents around the blackboard
  const cx = 10.15, cy = 3.75, R = 1.85, d = 1.0;
  s.addShape(pres.shapes.OVAL, { x: cx - 1.05, y: cy - 1.05, w: 2.1, h: 2.1, fill: { color: "1E293B" }, line: { color: "475569", width: 1.5, dashType: "dash" }, objectName: "blackboard" });
  s.addText("shared\nblackboard", { x: cx - 1.0, y: cy - 0.45, w: 2.0, h: 0.9, fontSize: 14, color: "CBD5E1", align: "center", valign: "middle", margin: 0, isTextBox: true });
  ["planner", "retriever", "processor", "critic", "archivist"].forEach((k, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
    agentDot(s, k, cx + R * Math.cos(a) - d / 2, cy + R * Math.sin(a) - d / 2, d);
    s.addText(AG[k].name, { x: cx + R * Math.cos(a) - 0.8, y: cy + R * Math.sin(a) + d / 2 + 0.02, w: 1.6, h: 0.3, fontSize: 12, color: "E2E8F0", align: "center", margin: 0, isTextBox: true });
  });
  s.addNotes(NOTES[1]);

  // ================================================================ 2
  pres.addSection({ title: "System" });
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "System" });
  s.addText("From a research goal to verified, stored evidence", { placeholder: "title" });
  const steps = [
    ["Goal", "plain-English research question", fa.FaUserGraduate, THEME.colors.dk2],
    ["Plan", "Planner decomposes into ≤ 4 subtasks", AG.planner.icon, AG.planner.hex],
    ["Retrieve", "arXiv · OpenAlex · Crossref with provenance", AG.retriever.icon, AG.retriever.hex],
    ["Process", "dedup · rank · atomic claims", AG.processor.icon, AG.processor.hex],
    ["Verify", "independent Critic checks each claim", AG.critic.icon, AG.critic.hex],
    ["Store", "report · BibTeX · vector index · trace", AG.archivist.icon, AG.archivist.hex],
  ];
  const sw = 1.85, sg = 0.2;
  for (let i = 0; i < steps.length; i++) {
    const [h, t, ic, col] = steps[i];
    const x = 0.6 + i * (sw + sg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: sw, h: 2.05, rectRadius: 0.12, fill: { color: "F8FAFC" }, line: { color: "E2E8F0" } });
    await iconCircle(s, ic, col, x + (sw - 0.75) / 2, 1.6, 0.75);
    s.addText(h, { x, y: 2.42, w: sw, h: 0.35, fontSize: 16, bold: true, color: col, align: "center", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.1, y: 2.78, w: sw - 0.2, h: 0.65, fontSize: 12, color: THEME.colors.dk2, align: "center", valign: "top", margin: 0, isTextBox: true });
    if (i < steps.length - 1) s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + sw + 0.03, y: 2.35, w: 0.14, h: 0.2, rotate: 90, fill: { color: "94A3B8" }, line: { color: "94A3B8" } });
  }
  // problem + criteria
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 3.85, w: 6.0, h: 2.65, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 } });
  s.addText([
    { text: "Why multi-agent?", options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
    { text: "Unaided LLMs plan unreliably and fabricate facts and citations, so each step gets its own specialist and every claim is checked.", options: { fontSize: 14, color: C.text2, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Brief success criteria → agents", options: { bold: true, fontSize: 14, color: C.text1, breakLine: true } },
    { text: "Identify & retrieve relevant data: Planner, Retriever", options: { bullet: true, fontSize: 13, color: C.text2, breakLine: true } },
    { text: "Process the data: Processor, Critic", options: { bullet: true, fontSize: 13, color: C.text2, breakLine: true } },
    { text: "Store in a usable format: Archivist", options: { bullet: true, fontSize: 13, color: C.text2 } },
  ], { x: 0.85, y: 3.98, w: 5.6, h: 2.45, valign: "top", margin: 0, paraSpaceAfter: 3, isTextBox: true });
  const stats = [["5", "agents implemented"], ["96", "automated tests · 92% coverage"], ["3/3", "live goals completed in budget"], ["~1 min", "per live run · ~21k tokens"]];
  stats.forEach(([n, l], i) => {
    const x = 6.95 + (i % 2) * 2.9, y = 3.85 + Math.floor(i / 2) * 1.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.75, h: 1.2, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: "CBD5E1" } });
    s.addText(n, { x: x + 0.15, y: y + 0.08, w: 2.45, h: 0.62, fontSize: 34, bold: true, color: C.accent1, fontFace: "Cambria", margin: 0, isTextBox: true });
    s.addText(l, { x: x + 0.15, y: y + 0.72, w: 2.5, h: 0.4, fontSize: 12, color: C.text2, margin: 0, isTextBox: true });
  });
  cite(s, "Kambhampati et al. (2024); Ji et al. (2023)");
  s.addNotes(NOTES[2]);

  // ================================================================ 3
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "System" });
  s.addText("Hybrid agents coordinating through a blackboard", { placeholder: "title" });
  // blackboard
  const bbx = 2.55, bby = 2.85, bbw = 3.55, bbh = 1.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bbx, y: bby, w: bbw, h: bbh, rectRadius: 0.12, fill: { color: "1E293B" }, line: { color: "1E293B" }, objectName: "blackboard" });
  s.addText([
    { text: "SQLite blackboard", options: { bold: true, fontSize: 16, color: "FFFFFF", breakLine: true } },
    { text: "runs · subtasks · sources · claims · verdicts · feedback · syntheses · trace", options: { fontSize: 12, color: "CBD5E1" } },
  ], { x: bbx + 0.15, y: bby + 0.1, w: bbw - 0.3, h: bbh - 0.2, align: "center", valign: "middle", margin: 0, isTextBox: true });
  const cards = [
    ["planner", 2.75, 1.4, "Deliberative · ReAct", "gpt-oss-120b"],
    ["retriever", 0.5, 2.85, "Reactive · no LLM", "3 open APIs"],
    ["processor", 6.3, 2.85, "Reactive", "gpt-oss-20b + MiniLM"],
    ["critic", 0.95, 4.95, "Reactive verifier", "qwen3.8-27b"],
    ["archivist", 4.55, 4.95, "Reactive · only disk writer", "no LLM"],
  ];
  for (const [k, x, y, role, model] of cards) {
    const w = k === "planner" ? 3.15 : 1.85, h = 1.25;
    const ww = (k === "critic" || k === "archivist") ? 3.15 : w;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: ww, h, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: AG[k].hex, width: 1.5 } });
    agentDot(s, k, x + 0.12, y + 0.15, 0.55);
    s.addText(AG[k].name, { x: x + 0.75, y: y + 0.12, w: ww - 0.85, h: 0.35, fontSize: 15, bold: true, color: AG[k].hex, margin: 0, isTextBox: true });
    s.addText([{ text: role, options: { breakLine: true } }, { text: model }], { x: x + 0.12, y: y + 0.72, w: ww - 0.2, h: 0.5, fontSize: 11, color: THEME.colors.dk2, margin: 0, isTextBox: true });
  }
  // connectors (read/write)
  const conn = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: "94A3B8", width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" } });
  conn(4.33, 2.65, 4.33, 2.85);
  conn(2.35, 3.48, 2.55, 3.48);
  conn(6.1, 3.48, 6.3, 3.48);
  conn(3.0, 4.6, 3.0, 4.95);
  conn(5.6, 4.6, 5.6, 4.95);
  // right column
  const pts = [
    [fa.FaLayerGroup, "Hybrid architecture", "One deliberative LLM planner; reactive specialists act only when their input appears (Wooldridge, 2009). Retrieval and reasoning fail on separate surfaces (Guo et al., 2024)."],
    [fa.FaTableCells, "Blackboard coordination", "Agents never call each other. Opportunistic control (Hayes-Roth, 1985); outperforms a central orchestrator for LLM information discovery (Salemi et al., 2025). Everything persisted, so it is explainable."],
    [fa.FaLock, "Safety by construction", "Atomic subtask claims in one IMMEDIATE transaction (8 racing threads → 1 winner). Gaia role permissions enforced at runtime (Wooldridge, Jennings and Kinny, 2000)."],
  ];
  for (let i = 0; i < pts.length; i++) {
    const [ic, h, t] = pts[i];
    const y = 1.45 + i * 1.75;
    await iconCircle(s, ic, THEME.colors.accent1, 8.45, y, 0.6);
    s.addText(h, { x: 9.2, y: y - 0.02, w: 3.5, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: 9.2, y: y + 0.36, w: 3.55, h: 1.3, fontSize: 12, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  }
  s.addNotes(NOTES[3]);

  // ================================================================ 4
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "System" });
  s.addText("Planning with ReAct, Reflexion and LLM-Modulo guardrails", { placeholder: "title" });
  // cycle
  const cyc = [
    ["Observation", "blackboard state: statuses, feedback, budget", fa.FaEye, 2.45, 1.4],
    ["Thought", "LLM reasoning, saved to the trace", fa.FaBrain, 4.45, 3.35],
    ["Action", "issue · refine · switch DB · abandon · synthesise", fa.FaBolt, 0.6, 3.35],
  ];
  s.addShape(pres.shapes.OVAL, { x: 2.15, y: 2.2, w: 2.9, h: 2.9, fill: { color: "FFFFFF" }, line: { color: "CBD5E1", width: 2, dashType: "dash" } });
  agentDot(s, "planner", 3.2, 3.15, 0.8);
  s.addText("Planner", { x: 2.8, y: 3.98, w: 1.6, h: 0.3, fontSize: 12, bold: true, color: AG.planner.hex, align: "center", margin: 0, isTextBox: true });
  for (const [h, t, ic, x, y] of cyc) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.3, h: 1.2, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: "CBD5E1" } });
    await iconCircle(s, ic, AG.planner.hex, x + 0.12, y + 0.14, 0.45);
    s.addText(h, { x: x + 0.65, y: y + 0.15, w: 1.6, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.12, y: y + 0.62, w: 2.1, h: 0.55, fontSize: 11, color: C.text2, margin: 0, isTextBox: true });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.25, w: 6.3, h: 1.25, rectRadius: 0.1, fill: { color: "FFF7ED" }, line: { color: AG.critic.hex } });
  s.addText([
    { text: "Reflexion-style re-planning  ", options: { bold: true, color: AG.critic.hex } },
    { text: "Retriever, Processor and Critic write natural-language Feedback when they reject work; the Planner must read it before its next Thought (Shinn et al., 2023).", options: { color: C.text2 } },
  ], { x: 0.8, y: 5.32, w: 5.95, h: 1.1, fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
  // guardrails
  s.addText("LLM-Modulo: the LLM proposes, code decides (Kambhampati et al., 2024)", { x: 7.3, y: 1.4, w: 5.45, h: 0.6, fontSize: 16, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  const guards = [
    [fa.FaListCheck, "Allowed actions computed by code each step"],
    [fa.FaCode, "JSON schema + semantic validation (no repeated failed query)"],
    [fa.FaRotateLeft, "Deterministic fallback policy after 2 invalid replies"],
    [fa.FaHourglassHalf, "20-step ReAct budget guarantees termination"],
    [fa.FaBan, "≤ 3 re-plans per subtask, then 'unresolvable' (no LLM call)"],
    [fa.FaFileShield, "Synthesis rejected twice → extractive verified-only fallback"],
  ];
  for (let i = 0; i < guards.length; i++) {
    const [ic, t] = guards[i];
    const y = 2.2 + i * 0.72;
    await iconCircle(s, ic, THEME.colors.dk1, 7.3, y, 0.48);
    s.addText(t, { x: 7.95, y: y - 0.02, w: 4.8, h: 0.52, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  }
  cite(s, "ReAct: Yao et al. (2023) · Reflexion: Shinn et al. (2023) · LLM-Modulo: Kambhampati et al. (2024) · re-plan cap: Unit 6 feedback #3");
  s.addNotes(NOTES[4]);

  // ================================================================ 5  UML class diagram
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "System" });
  s.addText("Structural view: UML class diagram (new in Unit 11)", { placeholder: "title" });
  const box = (x, y, w, h, name, stereo, body, head = THEME.colors.dk1) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "FFFFFF" }, line: { color: head, width: 1.25 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.42, fill: { color: head }, line: { color: head } });
    s.addText(stereo ? [{ text: stereo + " ", options: { italic: true, fontSize: 10 } }, { text: name, options: { bold: true, fontSize: 13 } }] : name,
      { x, y, w, h: 0.42, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    if (body) s.addText(body, { x: x + 0.08, y: y + 0.45, w: w - 0.16, h: h - 0.5, fontSize: 10.5, color: THEME.colors.dk2, valign: "top", margin: 0, fontFace: "Calibri", isTextBox: true });
  };
  const line = (x1, y1, x2, y2, opt = {}) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0, h: Math.abs(y2 - y1) || 0, line: { color: "475569", width: 1.25, ...opt } });
  // band A
  box(0.6, 1.35, 2.4, 0.95, "Controller", "", "+run(goal) · +step()\nruns first agent that can_act");
  box(3.75, 1.35, 2.7, 0.95, "Agent", "«abstract»", "+role · +can_act(bb) · +act(bb)");
  box(7.2, 1.35, 2.6, 0.95, "ScopedBlackboard", "", "enforces PERMISSIONS[role]");
  box(10.4, 1.35, 2.35, 0.95, "Blackboard", "", "SQLite · claim_subtask()\nadd_feedback() · log()");
  line(3.0, 1.82, 3.75, 1.82, { beginArrowType: "diamond" });
  s.addText("1..5", { x: 3.3, y: 1.55, w: 0.45, h: 0.25, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  line(6.45, 1.82, 7.2, 1.82, { endArrowType: "arrow", dashType: "dash" });
  s.addText("uses", { x: 6.6, y: 1.55, w: 0.5, h: 0.25, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  line(9.8, 1.82, 10.4, 1.82, { endArrowType: "arrow" });
  // band B: subclasses
  const subs = [["planner", "PlannerAgent", "_react_step() · _validate()\n⇢ LLMClient"], ["retriever", "RetrieverAgent", "search 3 APIs\n⇢ API clients"], ["processor", "ProcessorAgent", "select_sources()\n⇢ LLMClient, Embedder"], ["critic", "CriticAgent", "claim + synthesis checks\n⇢ LLMClient"], ["archivist", "ArchivistAgent", "report · BibTeX · index\n⇢ Embedder"]];
  const bw = 2.3, bg = 0.15, by = 3.0;
  // UML generalisation uses a hollow triangle; pptx arrowheads are filled, so draw it as a shape.
  line(5.1, 2.44, 5.1, 2.62);
  s.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: 5.0, y: 2.3, w: 0.2, h: 0.15, fill: { color: "FFFFFF" }, line: { color: "475569", width: 1.25 }, objectName: "generalisation arrowhead" });
  line(0.6 + bw / 2, 2.62, 0.6 + 4 * (bw + bg) + bw / 2, 2.62);
  subs.forEach(([k, n, b], i) => {
    const x = 0.6 + i * (bw + bg);
    line(x + bw / 2, 2.62, x + bw / 2, by);
    box(x, by, bw, 0.95, n, "", b, AG[k].hex);
  });
  s.addText("generalisation", { x: 5.3, y: 2.32, w: 1.2, h: 0.25, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  // band C: packages
  const pkg = (x, w, label) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.55, w, h: 2.0, fill: { color: "F8FAFC" }, line: { color: "CBD5E1", dashType: "dash" } });
    s.addText(label, { x: x + 0.08, y: 4.57, w: w - 0.1, h: 0.25, fontSize: 10, italic: true, color: MUTED, margin: 0, isTextBox: true });
  };
  pkg(0.6, 4.6, "llm/");
  box(0.75, 4.85, 2.0, 0.75, "LLMClient", "«interface»", "+complete(task, model…)");
  box(2.95, 4.85, 2.1, 0.75, "OpenAICompatible", "", "Groq · fallbacks");
  box(0.75, 5.7, 2.0, 0.75, "MockLLM", "", "deterministic rules");
  box(2.95, 5.7, 2.1, 0.75, "ScriptedLLM", "", "forces failure paths");
  pkg(5.4, 2.4, "embeddings.py");
  box(5.55, 4.85, 2.1, 0.75, "Embedder", "«interface»", "+embed(texts)");
  box(5.55, 5.7, 2.1, 0.75, "MiniLM · Hashing", "", "local · offline tests");
  pkg(8.0, 3.2, "sources/");
  box(8.15, 4.85, 2.9, 0.75, "ScholarlyHTTP", "", "rate limits · cache · retries");
  box(8.15, 5.7, 2.9, 0.75, "Arxiv · OpenAlex · Crossref", "", "parse → common record");
  box(11.4, 4.85, 1.35, 1.6, "Settings", "«frozen»", "all thresholds, budgets, models");
  // dependencies (dashed)
  s.addText("◆ composition (Controller creates and owns its 5 agents)   △ generalisation (5 roles inherit Agent)   ⇢ dependency (listed in each class)", { x: 0.6, y: 6.68, w: 12.1, h: 0.3, fontSize: 10, italic: true, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes(NOTES[5]);

  // ================================================================ 6  Demonstration (screenshot walkthrough)
  // Four moments from real live runs, captured from the dashboard
  // (presentation/assets, produced by make_demo_assets.js).
  pres.addSection({ title: "Demonstration" });
  s = pres.addSlide({ masterName: "Content Dark", sectionTitle: "Demonstration" });
  s.addText("Demonstration: live runs, step by step", { placeholder: "title" });
  const ASSETS = path.join(ROOT, "presentation", "assets");
  const panels = [
    ["planner", "demo_1_plan.png", "Planner reasons, then acts: Thought → create_plan → issue_subtask"],
    ["critic", "demo_2_critic.png", "Critic rejects the planted claim, citing the real abstract (16/17 kept)"],
    ["processor", "demo_3_replan.png", "Feedback drives re-planning: 2 re-plans, then 4/4 verified (earlier live run)"],
    ["archivist", "demo_4_report.png", "Verified report: Harvard citations keep claim IDs; BibTeX and trace saved"],
  ];
  const PW = 5.95, PH = 2.15;
  for (let i = 0; i < panels.length; i++) {
    const [k, file, caption] = panels[i];
    const px = 0.6 + (i % 2) * (PW + 0.2);
    const cy = 1.3 + Math.floor(i / 2) * 2.65;
    const py = cy + 0.36;
    // caption: numbered badge in the agent's colour
    s.addShape(pres.shapes.OVAL, { x: px, y: cy + 0.02, w: 0.3, h: 0.3, fill: { color: AG[k].hex }, line: { color: AG[k].hex } });
    s.addText(String(i + 1), { x: px, y: cy + 0.02, w: 0.3, h: 0.3, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(caption, { x: px + 0.4, y: cy, w: PW - 0.4, h: 0.34, fontSize: 13, bold: true, color: "E2E8F0", valign: "middle", margin: 0, isTextBox: true });
    // screenshot, fitted inside a dark frame without distortion
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: PW, h: PH, fill: { color: "0E1117" }, line: { color: "334155", width: 1 }, objectName: `panel ${i + 1} frame` });
    const meta = await sharp(path.join(ASSETS, file)).metadata();
    const ratio = meta.width / meta.height;
    const pad = 0.06;
    let w = PW - 2 * pad, h = w / ratio;
    if (h > PH - 2 * pad) { h = PH - 2 * pad; w = h * ratio; }
    s.addImage({ path: path.join(ASSETS, file), x: px + (PW - w) / 2, y: py + (PH - h) / 2, w, h, objectName: `panel ${i + 1} screenshot` });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.56, w: 12.1, h: 0.42, rectRadius: 0.06, fill: { color: "1E293B" }, line: { color: "334155" } });
  s.addText([
    { text: "All evidence → ", options: { bold: true, color: THEME.colors.accent2 } },
    { text: "github.com/adilaljasmi/scholarplan", options: { bold: true, color: "FFFFFF", hyperlink: { url: "https://github.com/adilaljasmi/scholarplan" } } },
    { text: "  ·  evidence/runs  ·  evidence/screenshots  ·  evidence/tests  ·  evidence/evaluation  ·  docs/REMEDIATION_LOG.md", options: { color: "CBD5E1" } },
  ], { x: 0.75, y: 6.56, w: 11.85, h: 0.42, fontSize: 11, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes(NOTES[6]);

  // ================================================================ 7  Evaluation
  pres.addSection({ title: "Evidence" });
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "Evidence" });
  s.addText("Does it work? Evaluation against measurable targets", { placeholder: "title" });
  s.addText("Critic on 21 hand-labelled claim–source pairs (9 hallucinations)", { x: 0.6, y: 1.35, w: 6.6, h: 0.35, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addChart(pres.charts.BAR, [
    { name: "Lexical-overlap baseline", labels: ["Precision", "Recall", "Hallucinations caught"], values: [0.83, 0.83, 0.78] },
    { name: "qwen3.8-27b (production)", labels: ["Precision", "Recall", "Hallucinations caught"], values: [1.0, 1.0, 1.0] },
    { name: "gpt-oss-120b (fallback)", labels: ["Precision", "Recall", "Hallucinations caught"], values: [1.0, 1.0, 1.0] },
  ], {
    x: 0.6, y: 1.75, w: 6.6, h: 3.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["94A3B8", AG.critic.hex, THEME.colors.accent1],
    valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: THEME.colors.dk2,
    catAxisLabelColor: THEME.colors.dk2, valAxisLabelColor: MUTED, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    valGridLine: { color: "E2E8F0", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: THEME.colors.dk2,
  });
  s.addText("Targets: precision ≥ 0.85, recall ≥ 0.75. Small benchmark: a perfect score likely reflects a ceiling effect.", { x: 0.6, y: 5.7, w: 6.6, h: 0.55, fontSize: 12, italic: true, color: C.text2, margin: 0, isTextBox: true });
  const evs = [
    ["0.90", "Retrieval recall (target ≥ 0.75)", "10 fixed queries for papers our Unit 6 report cited; arXiv 8 · OpenAlex 6 · Crossref 1. Miss: Hayes-Roth (1985), no indexed abstract."],
    ["3/3", "Live goals completed within budget", "Mean 6.7 ReAct steps, 58.5 s and ~21k tokens per run; 2 genuine re-plans."],
    ["0.92 · 0.30", "Dedup / relevance cosine thresholds", "Named constants, unit-tested both ways; top-3 sources, ≤ 2 claims each, pass if ≥ 50% supported."],
  ];
  evs.forEach(([n, h, t], i) => {
    const y = 1.35 + i * 1.72;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.6, y, w: 5.15, h: 1.55, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(n, { x: 7.8, y: y + 0.1, w: 2.1, h: 0.75, fontSize: 30, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: 9.95, y: y + 0.12, w: 2.7, h: 0.7, fontSize: 13, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 7.8, y: y + 0.88, w: 4.85, h: 0.62, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  cite(s, "Responds to Unit 6 feedback #2 (measurable targets). Raw results: evidence/evaluation/*.json · eval/run_eval.py");
  s.addNotes(NOTES[7]);

  // ================================================================ 8  Testing
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "Evidence" });
  s.addText("Testing, remediation and development progress", { placeholder: "title" });
  const tstats = [["96", "tests passing"], ["92%", "line coverage"], ["4/4", "live smoke tests"], ["20", "defects remediated"]];
  tstats.forEach(([n, l], i) => {
    const x = 0.6 + (i % 2) * 1.95, y = 1.4 + Math.floor(i / 2) * 1.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.8, h: 1.15, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(n, { x, y: y + 0.08, w: 1.8, h: 0.6, fontSize: 30, bold: true, color: C.accent1, fontFace: "Cambria", align: "center", margin: 0, isTextBox: true });
    s.addText(l, { x, y: y + 0.7, w: 1.8, h: 0.35, fontSize: 12, color: C.text2, align: "center", margin: 0, isTextBox: true });
  });
  // layers
  const layers = [
    ["Evaluation harness", "quality of LLM parts vs targets", 1.6, "0E7C7B"],
    ["Live smoke (4)", "API / model drift", 2.2, "16A34A"],
    ["Functional (12)", "whole system, forced failure paths", 2.8, "2563EB"],
    ["Unit (84)", "blackboard · parsers · rules · formatting", 3.4, "1E293B"],
  ];
  layers.forEach(([h, t, w, col], i) => {
    const y = 4.15 + i * 0.6;
    const x = 0.6 + (3.75 - w * 1.1) / 2;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: w * 1.1, h: 0.52, fill: { color: col }, line: { color: col } });
    s.addText(h, { x, y, w: w * 1.1, h: 0.52, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Scripted mock LLM forces rejection, re-planning and budget paths deterministically", { x: 0.6, y: 6.55, w: 4.0, h: 0.4, fontSize: 10, italic: true, color: MUTED, margin: 0, isTextBox: true });
  // remediation cards
  s.addText("Remediation highlights (docs/REMEDIATION_LOG.md)", { x: 4.9, y: 1.35, w: 7.85, h: 0.4, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const rems = [
    [fa.FaTriangleExclamation, "R8 · Critic model withdrawn", "Groq 404 for Llama-3.3-70B → Qwen (other family), automatic model substitution, regression test."],
    [fa.FaGaugeHigh, "R10 · API rate limiting", "OpenAlex 429 in the first live run → honour Retry-After, exponential back-off, 1 req/s spacing."],
    [fa.FaBug, "R15 · Planner misdiagnosis", "Model format error read as a 'bad query'; a good query was rewritten. Feedback now states the cause. Lesson: self-correction is only as good as its feedback."],
  ];
  for (let i = 0; i < rems.length; i++) {
    const [ic, h, t] = rems[i];
    const y = 1.85 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.9, y, w: 7.85, h: 1.02, rectRadius: 0.08, fill: { color: "FFFFFF" }, line: { color: "E2E8F0" } });
    await iconCircle(s, ic, AG.critic.hex, 5.05, y + 0.22, 0.58);
    s.addText(h, { x: 5.8, y: y + 0.06, w: 6.8, h: 0.35, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: 5.8, y: y + 0.4, w: 6.85, h: 0.58, fontSize: 12, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  }
  // progress timeline
  s.addText("Development progress (Git history)", { x: 4.9, y: 5.4, w: 7.85, h: 0.3, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const tl = ["Unit 6 design", "Scaffold", "Blackboard & APIs", "LLM layer & agents", "Unit + functional tests", "Evaluation", "Dashboard & live eval", "README & evidence", "Presentation"];
  const tx0 = 5.0, tx1 = 12.6;
  s.addShape(pres.shapes.LINE, { x: tx0, y: 5.95, w: tx1 - tx0, h: 0, line: { color: "CBD5E1", width: 2 } });
  tl.forEach((t, i) => {
    const x = tx0 + (i * (tx1 - tx0)) / (tl.length - 1);
    s.addShape(pres.shapes.OVAL, { x: x - 0.09, y: 5.86, w: 0.18, h: 0.18, fill: { color: i === 0 ? THEME.colors.accent2 : THEME.colors.accent1 }, line: { color: "FFFFFF" } });
    s.addText(t, { x: x - 0.46, y: 6.1, w: 0.92, h: 0.5, fontSize: 10, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(NOTES[8]);

  // ================================================================ 9  Critical reflection
  pres.addSection({ title: "Reflection" });
  s = pres.addSlide({ masterName: "Content Light", sectionTitle: "Reflection" });
  s.addText("Critical reflection: deviations, limitations and ethics", { placeholder: "title" });
  const cols = [
    [fa.FaCodeBranch, "Deviations, and why", THEME.colors.accent1, [
      ["Azure GPT-5-mini → Groq gpt-oss", "no Azure subscription; OpenAI-compatible API, so moving back is a one-line change"],
      ["Llama-3.3-70B → Qwen Critic", "model withdrawn; the design's real need, an independent model family, is preserved"],
      ["Ollama minicheck → scripted mocks", "no GPU; mocks force failure paths deterministically"],
    ]],
    [fa.FaMagnifyingGlassChart, "Limitations", AG.critic.hex, [
      ["Abstract-level verification", "the full paper may qualify a claim the abstract states"],
      ["Lenient on connective prose", "one uncited bridging sentence passed the synthesis check"],
      ["Small Critic benchmark (n = 21)", "a perfect score is likely a ceiling effect"],
    ]],
    [fa.FaScaleBalanced, "Legal · social · ethical · professional", AG.planner.hex, [
      ["Terms of use", "arXiv 3 s rule, identified client, caching"],
      ["Copyright & attribution", "metadata and abstracts only; every claim cited"],
      ["Misinformation & integrity", "rejected claims excluded; AI-generated notice; a reading aid, not a substitute"],
      ["Bias, privacy & security", "English/recency bias stated; goal sent to a third party; keys git-ignored; paths scrubbed"],
    ]],
  ];
  for (let c = 0; c < cols.length; c++) {
    const [ic, h, col, items] = cols[c];
    const x = 0.6 + c * 4.1, w = 3.9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.35, w, h: 3.95, rectRadius: 0.1, fill: { color: "F8FAFC" }, line: { color: "E2E8F0" } });
    await iconCircle(s, ic, col, x + 0.2, 1.52, 0.6);
    s.addText(h, { x: x + 0.95, y: 1.5, w: w - 1.05, h: 0.65, fontSize: 16, bold: true, color: col, valign: "middle", margin: 0, isTextBox: true });
    const runs = [];
    items.forEach(([a, b], i) => {
      runs.push({ text: a, options: { bold: true, fontSize: 13, color: C.text1, breakLine: true } });
      runs.push({ text: b, options: { fontSize: 12, color: C.text2, breakLine: i < items.length - 1, paraSpaceAfter: 9 } });
    });
    s.addText(runs, { x: x + 0.2, y: 2.3, w: w - 0.4, h: 2.95, valign: "top", margin: 0, isTextBox: true });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.5, w: 12.1, h: 1.05, rectRadius: 0.1, fill: { color: THEME.colors.dk1 }, line: { color: THEME.colors.dk1 } });
  s.addText("Next steps", { x: 0.85, y: 5.5, w: 1.8, h: 1.05, fontSize: 16, bold: true, color: THEME.colors.accent2, fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
  const nexts = [[fa.FaFileLines, "Open-access full text with passage-level checking"], [fa.FaQuoteRight, "Sentence-level citation rule for the synthesis"], [fa.FaFlask, "Larger Critic benchmark (LLM-AggreFact scale)"]];
  for (let i = 0; i < nexts.length; i++) {
    const x = 2.7 + i * 3.35;
    await iconCircle(s, nexts[i][0], THEME.colors.accent1, x, 5.75, 0.55);
    s.addText(nexts[i][1], { x: x + 0.68, y: 5.6, w: 2.55, h: 0.85, fontSize: 13, color: "E2E8F0", valign: "middle", margin: 0, isTextBox: true });
  }
  s.addNotes(NOTES[9]);

  // ================================================================ 10 Conclusion + references
  s = pres.addSlide({ masterName: "Content Dark", sectionTitle: "Reflection" });
  s.addText("Conclusion", { placeholder: "title" });
  const takes = [
    ["Design met", "Hybrid ReAct planner + reactive specialists on a SQLite blackboard, as proposed in Unit 6"],
    ["Hallucinations filtered", "Independent Critic: P = R = 1.00 on the benchmark; caught a planted claim in a live run"],
    ["Feedback addressed", "Class diagram · measurable targets · 3-re-plan cap, all implemented and tested"],
  ];
  takes.forEach(([h, t], i) => {
    const y = 1.4 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.08, w: 0.42, h: 0.42, fill: { color: THEME.colors.accent2 }, line: { color: THEME.colors.accent2 } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: THEME.colors.dk1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: h, options: { bold: true, fontSize: 17, color: "FFFFFF", breakLine: true } }, { text: t, options: { fontSize: 13, color: "CBD5E1" } }],
      { x: 1.2, y, w: 5.3, h: 1.0, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText("“Let the LLM supply judgement; let code guarantee validity and termination.”", { x: 0.6, y: 4.85, w: 5.9, h: 0.9, fontSize: 18, italic: true, color: THEME.colors.accent2, fontFace: "Cambria", margin: 0, isTextBox: true });
  s.addText("github.com/adilaljasmi/scholarplan · code · tests · evidence · README", { x: 0.6, y: 5.9, w: 5.9, h: 0.35, fontSize: 12, color: "94A3B8", margin: 0, isTextBox: true });
  const refs = [
    "Guo, T. et al. (2024) 'Large language model based multi-agents: a survey of progress and challenges', IJCAI-24, pp. 8048–8057.",
    "Hayes-Roth, B. (1985) 'A blackboard architecture for control', Artificial Intelligence, 26(3), pp. 251–321.",
    "Ji, Z. et al. (2023) 'Survey of hallucination in natural language generation', ACM Computing Surveys, 55(12), pp. 1–38.",
    "Kambhampati, S. et al. (2024) 'Position: LLMs can't plan, but can help planning in LLM-Modulo frameworks', ICML 2024. PMLR 235.",
    "Salemi, A. et al. (2025) 'LLM-based multi-agent blackboard system for information discovery in data science', arXiv:2510.01285.",
    "Shinn, N. et al. (2023) 'Reflexion: language agents with verbal reinforcement learning', NeurIPS 36.",
    "Wooldridge, M. (2009) An Introduction to MultiAgent Systems. 2nd edn. Chichester: Wiley.",
    "Wooldridge, M., Jennings, N.R. and Kinny, D. (2000) 'The Gaia methodology for agent-oriented analysis and design', AAMAS, 3(3), pp. 285–312.",
    "Yao, S. et al. (2023) 'ReAct: synergizing reasoning and acting in language models', ICLR 2023.",
  ];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 1.35, w: 5.85, h: 4.6, rectRadius: 0.08, fill: { color: "1E293B" }, line: { color: "334155" } });
  s.addText("References", { x: 7.1, y: 1.45, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText(refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 4 } })),
    { x: 7.1, y: 1.9, w: 5.5, h: 3.95, fontSize: 10.5, color: "CBD5E1", valign: "top", margin: 0, isTextBox: true });
  s.addText("Full Harvard reference list with DOIs and access dates: README §13", { x: 6.9, y: 6.05, w: 5.85, h: 0.3, fontSize: 10, italic: true, color: "94A3B8", margin: 0, isTextBox: true });
  s.addNotes(NOTES[10]);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
