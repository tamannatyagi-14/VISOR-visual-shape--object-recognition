# ─── VISOR — config.py ────────────────────────────────
# Visual Shape & Object Recognition
# All settings in one place — change here, works everywhere!
# ──────────────────────────────────────────────────────

# ── Shape Detection Settings ──────────────────────────
MIN_AREA        = 500    # minimum shape size in pixels
MATCH_THRESHOLD = 0.30    # lower = stricter match

# ── Color Settings (HSV ranges) ───────────────────────
TARGET_COLORS = {
    "white": ([0,   0,   150], [180, 40,  255]),
    "red":   ([0,   100, 100], [10,  255, 255]),
    "gold":  ([15,  80,  100], [35,  255, 255]),
}
# ── Webcam Settings ───────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 600

# ── UI Theme — Black Gold ─────────────────────────────
THEME = {
    "bg":        (10,  8,   0),    # background
    "accent":    (55,  175, 212),  # gold — BGR
    "accent_dim":(16,  80,  107),  # dark gold — BGR
    "text":      (55,  175, 212),  # text color
    "black":     (0,   0,   0),
}

# ── Detection Settings ────────────────────────────────
GLOW_FRAMES   = 20     # number of frames to glow detected shapes
MIN_NESTED    = 3    # minimum nested squares count