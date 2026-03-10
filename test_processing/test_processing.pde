import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress pythonAddr;   // Python OSC command receiver (port 9001)

String loadedFile = "";  // last file sent to Python

// Raw OSC values
float temporal  = 0;
float parietal  = 0;
float frontal   = 0;
float occipital = 0;

// Smoothed display values
float tSmooth = 0;
float pSmooth = 0;
float fSmooth = 0;
float oSmooth = 0;

boolean isPlaying = false;

// Lobe objects
Lobe lobeFrontal;
Lobe lobeTemporal;
Lobe lobeParietal;
Lobe lobeOccipital;

// ─────────────────────────────────────────────────
// Lobe class
// ─────────────────────────────────────────────────
class Lobe {
  float x, y, baseR;
  String label;
  color col;

  static final int N = 55;      // number of sprinkle particles
  float[] pAngle = new float[N];
  float[] pFrac  = new float[N]; // fraction of radius (resting position)
  float[] pSize  = new float[N];

  Lobe(float x, float y, float r, String lbl, color c, int seed) {
    this.x = x; this.y = y;
    this.baseR = r;
    this.label = lbl;
    this.col   = c;
    randomSeed(seed);
    for (int i = 0; i < N; i++) {
      pAngle[i] = random(TWO_PI);
      pFrac[i]  = random(0.12, 0.88);
      pSize[i]  = random(2.5, 7.0);
    }
  }

  void draw(float act) {
    float r = baseR + 28 * act;
    float cr = red(col), cg = green(col), cb = blue(col);

    // ── glow rings (outermost to innermost) ──────────────────
    noStroke();
    for (int g = 5; g >= 1; g--) {
      float gr = r + g * 20 * act;
      fill(cr, cg, cb, 12 * act * (6 - g));
      ellipse(x, y, gr * 2, gr * 2);
    }

    // ── main circle body ─────────────────────────────────────
    fill(cr * 0.25, cg * 0.25, cb * 0.25, 210);
    stroke(cr, cg, cb, 160 + 95 * act);
    strokeWeight(1.5 + 3.5 * act);
    ellipse(x, y, r * 2, r * 2);
    noStroke();

    // ── inner bright centre (hot-spot when active) ────────────
    float centreR = r * 0.28 * act;
    if (centreR > 1) {
      fill(cr, cg, cb, 60 * act);
      ellipse(x, y, centreR * 2, centreR * 2);
    }

    // ── sprinkle particles ────────────────────────────────────
    for (int i = 0; i < N; i++) {
      // particles spread outward and jitter with activation
      float jitter = act > 0.05 ? random(-0.06, 0.06) : 0;
      float dist   = pFrac[i] * r * (0.25 + 0.75 * act + jitter);
      float px     = x + cos(pAngle[i]) * dist;
      float py     = y + sin(pAngle[i]) * dist;
      float sz     = pSize[i] * (0.4 + 0.9 * act);
      float alpha  = 60 + 195 * act * pFrac[i]; // deeper particles are dimmer
      fill(cr * 0.6 + 100 * act, cg * 0.6 + 60 * act, cb * 0.6, alpha);
      ellipse(px, py, sz, sz);
    }

    // ── label ─────────────────────────────────────────────────
    textAlign(CENTER, TOP);
    fill(255, 180 + 75 * act);
    textSize(14);
    text(label, x, y + r + 10);

    // ── activation value ──────────────────────────────────────
    fill(200, 120 + 135 * act);
    textSize(11);
    text(nf(act, 1, 2), x, y + r + 27);
  }
}

// ─────────────────────────────────────────────────
// Setup & Draw
// ─────────────────────────────────────────────────
void setup() {
  size(900, 650);
  smooth(4);
  oscP5     = new OscP5(this, 9000);                  // receive analysis from Python
  pythonAddr = new NetAddress("127.0.0.1", 9001);     // send commands to Python

  // Layout: diamond — frontal top, temporal left, parietal right, occipital bottom
  float cx = width / 2.0;
  float cy = 360;
  lobeFrontal   = new Lobe(cx,       cy - 175, 72, "Frontal",   color(230,  75,  75), 1);
  lobeTemporal  = new Lobe(cx - 210, cy +  15, 68, "Temporal",  color(160,  60, 220), 2);
  lobeParietal  = new Lobe(cx + 210, cy +  15, 68, "Parietal",  color( 50, 200, 110), 3);
  lobeOccipital = new Lobe(cx,       cy + 175, 65, "Occipital", color( 70,  90, 230), 4);
}

void draw() {
  background(15);
  drawUI();
  smoothValues();
  drawConnectors();
  drawLobes();
}

// ─────────────────────────────────────────────────
// Thin connector lines between lobes
// ─────────────────────────────────────────────────
void drawConnectors() {
  strokeWeight(1);
  Lobe[] lobes = { lobeFrontal, lobeTemporal, lobeParietal, lobeOccipital };
  float[] acts = { fSmooth, tSmooth, pSmooth, oSmooth };
  for (int i = 0; i < lobes.length; i++) {
    for (int j = i + 1; j < lobes.length; j++) {
      float combined = (acts[i] + acts[j]) * 0.5;
      stroke(255, 255, 255, 20 + 40 * combined);
      line(lobes[i].x, lobes[i].y, lobes[j].x, lobes[j].y);
    }
  }
  noStroke();
}

void drawLobes() {
  lobeFrontal.draw(fSmooth);
  lobeTemporal.draw(tSmooth);
  lobeParietal.draw(pSmooth);
  lobeOccipital.draw(oSmooth);
}

// ─────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────
void drawUI() {
  // title
  fill(220);
  textAlign(LEFT, BASELINE);
  textSize(15);
  text("EEG-AR  ·  Brain Lobe Visualizer", 20, 28);

  // port info
  textSize(11);
  fill(80, 200, 80);
  text("← :9000  → :9001", width - 120, 28);

  // buttons
  drawButton(20,  42, 110, 36, "LOAD",  color(60, 60, 70));
  drawButton(145, 42, 110, 36, "PLAY",  color(40, 90, 50));
  drawButton(270, 42, 110, 36, "STOP",  color(90, 40, 40));

  // loaded file name (trimmed)
  if (loadedFile.length() > 0) {
    String fname = loadedFile.substring(loadedFile.lastIndexOf('/') + 1);
    fill(160, 160, 180);
    textSize(11);
    text("  " + (isPlaying ? "▶  " : "■  ") + fname, 395, 65);
  } else {
    fill(100);
    textSize(11);
    text("  No file loaded — press LOAD", 395, 65);
  }

  // divider
  stroke(50);
  strokeWeight(1);
  line(0, 88, width, 88);
  noStroke();
}

void drawButton(int x, int y, int w, int h, String lbl, color bg) {
  fill(bg);
  rect(x, y, w, h, 6);
  fill(230);
  textAlign(CENTER, CENTER);
  textSize(13);
  text(lbl, x + w/2, y + h/2);
  textAlign(LEFT, BASELINE);
}

// ─────────────────────────────────────────────────
// Interaction
// ─────────────────────────────────────────────────
void mousePressed() {
  if (overButton(20,  42, 110, 36)) selectInput("Select audio file:", "fileSelected");
  if (overButton(145, 42, 110, 36)) {
    OscMessage m = new OscMessage("/brain/play");
    oscP5.send(m, pythonAddr);
    isPlaying = true;
  }
  if (overButton(270, 42, 110, 36)) {
    OscMessage m = new OscMessage("/brain/stop");
    oscP5.send(m, pythonAddr);
    isPlaying = false;
  }
}

boolean overButton(int x, int y, int w, int h) {
  return mouseX > x && mouseX < x+w && mouseY > y && mouseY < y+h;
}

void fileSelected(File selection) {
  if (selection == null) return;
  loadedFile = selection.getAbsolutePath();
  OscMessage m = new OscMessage("/brain/load");
  m.add(loadedFile);
  oscP5.send(m, pythonAddr);
  isPlaying = false;
}

// ─────────────────────────────────────────────────
// Value smoothing
// ─────────────────────────────────────────────────
void smoothValues() {
  float alpha = 0.15;
  tSmooth += (temporal  - tSmooth) * alpha;
  pSmooth += (parietal  - pSmooth) * alpha;
  fSmooth += (frontal   - fSmooth) * alpha;
  oSmooth += (occipital - oSmooth) * alpha;
}

// ─────────────────────────────────────────────────
// OSC
// ─────────────────────────────────────────────────
void oscEvent(OscMessage msg) {
  if (msg.checkAddrPattern("/brain/lobes")) {
    temporal  = msg.get(0).floatValue();
    parietal  = msg.get(1).floatValue();
    frontal   = msg.get(2).floatValue();
    occipital = msg.get(3).floatValue();
  }
}

// ─────────────────────────────────────────────────
// Cleanup
// ─────────────────────────────────────────────────
void stop() {
  super.stop();
}
