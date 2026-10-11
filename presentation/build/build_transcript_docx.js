// Builds presentation/ScholarPlan_Narration_Transcript.docx from TRANSCRIPT.md
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, HeadingLevel } = require("docx");

const ROOT = path.resolve(__dirname, "..");
const md = fs.readFileSync(path.join(ROOT, "TRANSCRIPT.md"), "utf8").split(/\r?\n/);

// Split a line into runs: *[stage directions]*, **bold**, *italic*, `code`.
function runs(text) {
  const out = [];
  const re = /(\*\[.+?\]\*|\*\*.+?\*\*|(?<!\w)\*[^*]+?\*(?!\w)|`.+?`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun(text.slice(last, m.index)));
    const t = m[0];
    if (t.startsWith("*[")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, color: "64748B" }));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas" }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun(text.slice(last)));
  return out;
}

const children = [];
let para = [];
const flush = () => {
  if (para.length) children.push(new Paragraph({ children: runs(para.join(" ")), spacing: { after: 160, line: 300 } }));
  para = [];
};
for (const raw of md) {
  const s = raw.trim();
  if (!s || s === "---") { flush(); continue; }
  const h = s.match(/^(#{1,3}) (.*)$/);
  if (h) {
    flush();
    const level = [HeadingLevel.TITLE, HeadingLevel.HEADING_1, HeadingLevel.HEADING_2][h[1].length - 1];
    children.push(new Paragraph({ heading: level, children: runs(h[2]), spacing: { before: 240, after: 120 } }));
    continue;
  }
  para.push(s);
}
flush();

const doc = new Document({
  creator: "Adil Aljasmi",
  title: "ScholarPlan presentation narration transcript",
  styles: {
    default: { document: { run: { font: "Calibri", size: 24 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: "Cambria", size: 40, bold: true, color: "0F172A" } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", run: { font: "Cambria", size: 28, bold: true, color: "0E7C7B" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", run: { font: "Cambria", size: 24, bold: true, color: "0F172A" } },
    ],
  },
  sections: [{ children }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "ScholarPlan_Narration_Transcript.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
