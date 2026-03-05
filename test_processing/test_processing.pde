import ddf.minim.*;
import oscP5.*;
import netP5.*;

Minim minim;
AudioPlayer player;

OscP5 oscP5;

float temporal = 0;
float parietal = 0;
float frontal = 0;
float occipital = 0;

float tSmooth = 0;
float pSmooth = 0;
float fSmooth = 0;
float oSmooth = 0;

boolean isPlaying = false;

void setup() {
  size(900, 600);
  minim = new Minim(this);
  oscP5 = new OscP5(this, 9000); // must match Python port
}

void draw() {
  background(20);

  drawUI();
  smoothValues();
  drawBrain();
}

void drawUI() {
  fill(255);
  textSize(16);
  text("Lobe Visualizer (Wav2Vec Proxy)", 20, 30);

  fill(80);
  rect(20, 50, 120, 40);
  fill(255);
  textAlign(CENTER, CENTER);
  text("LOAD", 80, 70);

  fill(80);
  rect(160, 50, 120, 40);
  fill(255);
  text("PLAY", 220, 70);

  fill(80);
  rect(300, 50, 120, 40);
  fill(255);
  text("STOP", 360, 70);

  textAlign(LEFT, BASELINE);
}

void mousePressed() {
  if (overButton(20,50,120,40)) {
    selectInput("Select audio file:", "fileSelected");
  }

  if (overButton(160,50,120,40) && player != null) {
    player.rewind();
    player.play();
    isPlaying = true;
  }

  if (overButton(300,50,120,40) && player != null) {
    player.pause();
    isPlaying = false;
  }
}

boolean overButton(int x, int y, int w, int h) {
  return mouseX > x && mouseX < x+w && mouseY > y && mouseY < y+h;
}

void fileSelected(File selection) {
  if (selection == null) return;

  if (player != null) {
    player.close();
  }

  player = minim.loadFile(selection.getAbsolutePath());
}

void smoothValues() {
  float alpha = 0.15;
  tSmooth += (temporal - tSmooth) * alpha;
  pSmooth += (parietal - pSmooth) * alpha;
  fSmooth += (frontal - fSmooth) * alpha;
  oSmooth += (occipital - oSmooth) * alpha;
}

void drawBrain() {

  float cx = width/2;
  float cy = height/2 + 50;

  noStroke();

  // Temporal (left)
  fill(100 + 155*tSmooth, 50, 200);
  ellipse(cx - 150, cy, 180 + 80*tSmooth, 140 + 50*tSmooth);

  // Parietal (top)
  fill(50, 200 + 55*pSmooth, 100);
  ellipse(cx, cy - 130, 200 + 60*pSmooth, 160 + 40*pSmooth);

  // Frontal (right)
  fill(200 + 55*fSmooth, 80, 80);
  ellipse(cx + 150, cy, 180 + 80*fSmooth, 140 + 50*fSmooth);

  // Occipital (back)
  fill(80, 80, 200 + 55*oSmooth);
  ellipse(cx - 50, cy + 100, 160 + 60*oSmooth, 120 + 40*oSmooth);

  drawLabels(cx, cy);
}

void drawLabels(float cx, float cy) {
  fill(255);
  textAlign(CENTER);
  text("Temporal", cx - 150, cy - 100);
  text("Parietal", cx, cy - 250);
  text("Frontal", cx + 150, cy - 100);
  text("Occipital", cx - 50, cy + 180);
}

void oscEvent(OscMessage msg) {
  if (msg.checkAddrPattern("/brain/lobes")) {
    temporal = msg.get(0).floatValue();
    parietal = msg.get(1).floatValue();
    frontal = msg.get(2).floatValue();
    occipital = msg.get(3).floatValue();
  }
}

void stop() {
  if (player != null) {
    player.close();
  }
  minim.stop();
  super.stop();
}
