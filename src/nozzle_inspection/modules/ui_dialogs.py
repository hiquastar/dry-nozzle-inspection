"""
Reusable UI dialogs for runtime overlays.
"""

import cv2


def confirm_action_dialog(window_name, frame, message1, message2, alpha=0.62):
    """Show a blocking yes/no overlay on the frame and return True (Y/space) or False (N/Q/Esc)."""
    while True:
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        box_w = min(w - 20, 980)
        box_h = min(h - 20, 120)
        x0 = max(10, int((w - box_w) * 0.5))
        y0 = max(10, int((h - box_h) * 0.5))
        x1 = min(w - 10, x0 + box_w)
        y1 = min(h - 10, y0 + box_h)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 255), 2)
        cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0, overlay)
        cv2.putText(
            overlay,
            str(message1),
            (x0 + 14, y0 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            str(message2),
            (x0 + 14, y0 + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
        )
        cv2.imshow(window_name, overlay)
        key_code = cv2.waitKeyEx(0)
        low_code = int(key_code & 0xFF)
        if low_code in (ord("y"), ord("Y"), 32):
            return True
        if low_code in (ord("n"), ord("N"), ord("q"), ord("Q"), 27):
            return False


def three_way_dialog(window_name, frame, title, line1, line2, line3, alpha=0.62):
    """3-option modal dialog.

    Returns one of: 'r' (replace), 'a' (append), 'c' (cancel).

    Keys:
        R / 1 / Y / SPACE -> replace
        A / 2             -> append
        C / N / Q / ESC   -> cancel
    """
    while True:
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        box_w = min(w - 20, 980)
        box_h = min(h - 20, 180)
        x0 = max(10, int((w - box_w) * 0.5))
        y0 = max(10, int((h - box_h) * 0.5))
        x1 = min(w - 10, x0 + box_w)
        y1 = min(h - 10, y0 + box_h)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 255), 2)
        cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0, overlay)
        cv2.putText(
            overlay,
            str(title),
            (x0 + 14, y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            str(line1),
            (x0 + 14, y0 + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (120, 255, 120),
            2,
        )
        cv2.putText(
            overlay,
            str(line2),
            (x0 + 14, y0 + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            overlay,
            str(line3),
            (x0 + 14, y0 + 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (180, 180, 180),
            2,
        )
        cv2.imshow(window_name, overlay)
        key_code = cv2.waitKeyEx(0)
        low_code = int(key_code & 0xFF)
        if low_code in (ord("r"), ord("R"), ord("1"), ord("y"), ord("Y"), 32):
            return "r"
        if low_code in (ord("a"), ord("A"), ord("2")):
            return "a"
        if low_code in (ord("c"), ord("C"), ord("n"), ord("N"), ord("q"), ord("Q"), 27):
            return "c"
