"""
Brawl Stars automation bot — BlueStacks (Windows).

Usage:
    python bot.py              # run the full loop
    python bot.py --dry-run    # print actions only, no input sent

Failsafe: move the mouse to the top-left corner of the screen to abort.
"""

import argparse
import json
import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

try:
    import mss
except ImportError:
    sys.exit("Install dependencies first: pip install -r requirements.txt")

from ocr import _get_reader

pyautogui.FAILSAFE = True

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Window capture
# ---------------------------------------------------------------------------

class WindowCapture:
    def __init__(self, title: str):
        self.title = title
        self._rect = None

    def _find(self):
        wins = gw.getWindowsWithTitle(self.title)
        if not wins:
            raise RuntimeError(f"Window not found: '{self.title}'")
        w = wins[0]
        w.activate()
        time.sleep(0.3)
        self._rect = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

    def focus(self):
        wins = gw.getWindowsWithTitle(self.title)
        if wins:
            wins[0].activate()
            time.sleep(0.3)

    @property
    def rect(self) -> dict:
        if self._rect is None:
            self._find()
        return self._rect

    def capture(self) -> np.ndarray:
        with mss.mss() as sct:
            raw = sct.grab(self.rect)
        img = np.array(raw)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def rel_to_abs(self, rx: int, ry: int) -> tuple[int, int]:
        r = self.rect
        return r["left"] + rx, r["top"] + ry

    def frac_to_abs(self, fx: float, fy: float) -> tuple[int, int]:
        r = self.rect
        return r["left"] + int(fx * r["width"]), r["top"] + int(fy * r["height"])


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

class StateDetector:
    def __init__(self, confidence: float = 0.4):
        self.confidence = confidence
        self._reader = _get_reader()

    def _scan(self, img: np.ndarray) -> list[tuple[str, float, tuple]]:
        """Return list of (text, confidence, center_xy) for all detections."""
        results = self._reader.readtext(img)
        out = []
        for (bbox, text, conf) in results:
            if conf >= self.confidence:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = int(sum(xs) / 4)
                cy = int(sum(ys) / 4)
                out.append((text.upper().strip(), conf, (cx, cy)))
        return out

    def find_text(self, img: np.ndarray, target: str) -> tuple[bool, tuple | None]:
        target = target.upper().strip()
        for text, _conf, center in self._scan(img):
            if target in text:
                return True, center
        return False, None

    def pixel_is_blue(self, img: np.ndarray, px: int, py: int) -> bool:
        """Return True if the pixel at (px, py) in the window screenshot is blue."""
        if py < 0 or py >= img.shape[0] or px < 0 or px >= img.shape[1]:
            return False
        hsv = cv2.cvtColor(img[py:py+1, px:px+1], cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        return 95 <= h <= 135 and s > 80 and v > 80

    def detect_game_end(self, img: np.ndarray) -> bool:
        """Fallback yellow-button detection used only when pixel coords are not configured."""
        h, w = img.shape[:2]
        x0, y0 = int(0.25 * w), int(0.8 * h)
        region = img[y0:h, x0:int(0.75 * w)]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20, 180, 180]), np.array([35, 255, 255]))
        region_size = region.shape[0] * region.shape[1]
        return (mask.sum() / 255) / region_size > 0.08

    def find_button_center(self, img: np.ndarray, color: str,
                           region_frac: tuple = (0.0, 0.7, 1.0, 1.0)) -> tuple | None:
        """Return pixel center of a colored button within a screen region fraction."""
        _ranges = {
            "yellow": ([20, 120, 120], [35, 255, 255]),
            "blue":   ([100, 120, 120], [130, 255, 255]),
            "green":  ([40, 100, 100], [80, 255, 255]),
        }
        if color not in _ranges:
            return None
        h, w = img.shape[:2]
        x0, y0 = int(region_frac[0] * w), int(region_frac[1] * h)
        x1, y1 = int(region_frac[2] * w), int(region_frac[3] * h)
        crop = img[y0:y1, x0:x1]
        lo, hi = np.array(_ranges[color][0]), np.array(_ranges[color][1])
        mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lo, hi)
        if mask.sum() < 3000:
            return None
        M = cv2.moments(mask)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]) + x0, int(M["m01"] / M["m00"]) + y0)
        return None

    def crop_region(self, img: np.ndarray, region: list[float]) -> np.ndarray:
        h, w = img.shape[:2]
        x0 = int(region[0] * w)
        y0 = int(region[1] * h)
        x1 = int(region[2] * w)
        y1 = int(region[3] * h)
        return img[y0:y1, x0:x1], (x0, y0)

    def find_lets_go(self, img: np.ndarray) -> tuple[bool, tuple | None]:
        """Detect the green rank-up LET'S GO button in the bottom-right region."""
        center = self.find_button_center(img, "green", region_frac=(0.5, 0.7, 1.0, 1.0))
        if center:
            return True, center
        return False, None


# ---------------------------------------------------------------------------
# Input controller
# ---------------------------------------------------------------------------

class InputController:
    def __init__(self, capture: WindowCapture, dry_run: bool = False):
        self.capture = capture
        self.dry_run = dry_run
        self._held: set[str] = set()

    def _log(self, msg: str):
        print(f"  [input] {msg}")

    def hold(self, key: str):
        if key in self._held:
            return
        self._log(f"keyDown({key!r})")
        if not self.dry_run:
            pyautogui.keyDown(key)
        self._held.add(key)

    def release(self, key: str):
        if key not in self._held:
            return
        self._log(f"keyUp({key!r})")
        if not self.dry_run:
            pyautogui.keyUp(key)
        self._held.discard(key)

    def press(self, key: str):
        self._log(f"press({key!r})")
        if not self.dry_run:
            pyautogui.press(key)

    def release_all(self):
        for key in list(self._held):
            self.release(key)

    def click_rel(self, rx: int, ry: int):
        ax, ay = self.capture.rel_to_abs(rx, ry)
        self._log(f"click(abs={ax},{ay}  rel={rx},{ry})")
        if not self.dry_run:
            pyautogui.click(ax, ay)

    def click_abs(self, ax: int, ay: int):
        self._log(f"click(abs={ax},{ay})")
        if not self.dry_run:
            pyautogui.click(ax, ay)

    def click_center(self, img_center_xy: tuple, window_offset_xy: tuple):
        ax = window_offset_xy[0] + img_center_xy[0]
        ay = window_offset_xy[1] + img_center_xy[1]
        self.click_abs(ax, ay)

    def swipe(self, from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.5):
        self._log(f"swipe({from_x},{from_y}) → ({to_x},{to_y})")
        if not self.dry_run:
            pyautogui.moveTo(from_x, from_y)
            pyautogui.dragTo(to_x, to_y, duration=duration, button="left")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    INITIAL_PLAY      = auto()
    DUELS_GAME_1      = auto()
    WAIT_PLAY_AGAIN   = auto()
    DUELS_GAME_2      = auto()
    NAVIGATE_MENU     = auto()
    SELECT_BB_BRAWLER = auto()
    NAVIGATE_BRAWLBALL= auto()
    BRAWLBALL_QUEUE   = auto()
    BRAWLBALL_GAME    = auto()
    NAVIGATE_MENU_2   = auto()
    RESTORE_BRAWLER   = auto()
    NAVIGATE_DUELS    = auto()


class GameBot:
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.capture = WindowCapture(cfg["window_title"])
        self.detector = StateDetector(confidence=cfg["ocr_confidence"])
        self.ctrl = InputController(self.capture, dry_run)
        self.state = State.INITIAL_PLAY
        self.poll = cfg["poll_interval_s"]
        self.autoaim_interval = cfg["autoaim_interval_s"]

    # --- helpers ------------------------------------------------------------

    def _log(self, msg: str):
        print(f"[{self.state.name}] {msg}")

    def _screenshot(self) -> np.ndarray:
        return self.capture.capture()

    def _wait_for_text(self, target: str, timeout: float = 120.0) -> tuple | None:
        """Poll until target text appears; returns its center in the window or None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            img = self._screenshot()
            found, center = self.detector.find_text(img, target)
            if found and center:
                # center is relative to window top-left already (screenshot coords)
                return center
            time.sleep(self.poll)
        return None

    def _click_text(self, target: str, timeout: float = 120.0) -> bool:
        """Wait for target text to appear, then click it."""
        self._log(f"Waiting for '{target}'...")
        center = self._wait_for_text(target, timeout)
        if center is None:
            self._log(f"Timeout waiting for '{target}'")
            return False
        r = self.capture.rect
        self.ctrl.click_abs(r["left"] + center[0], r["top"] + center[1])
        time.sleep(0.3)
        return True

    def _click_color_button(self, color: str, timeout: float = 30.0,
                            region_frac: tuple = (0.0, 0.7, 1.0, 1.0)) -> bool:
        """Poll until a colored button appears in region, then click its center."""
        self._log(f"Waiting for {color} button...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            img = self._screenshot()
            center = self.detector.find_button_center(img, color, region_frac=region_frac)
            if center:
                r = self.capture.rect
                self.ctrl.click_abs(r["left"] + center[0], r["top"] + center[1])
                time.sleep(0.4)
                return True
            time.sleep(0.2)
        self._log(f"Timeout waiting for {color} button")
        return False

    # --- states -------------------------------------------------------------

    def _state_initial_play(self):
        self._log("Clicking PLAY — waiting 14s for game to load")
        ax, ay = self.cfg["home_button"]
        self.ctrl.click_abs(ax, ay)
        time.sleep(14)

    def _state_duels_game(self, game_num: int):
        self._log(f"Duels game {game_num} started — holding {self.cfg['move_key']!r}")
        pixel = self.cfg.get("duels_ingame_pixel")
        self.ctrl.hold(self.cfg["move_key"])
        while True:
            img = self._screenshot()
            ended = (not self.detector.pixel_is_blue(img, pixel[0], pixel[1])
                     if pixel else self.detector.detect_game_end(img))
            if ended:
                self._log("Game ended")
                break
            time.sleep(self.poll)
        self.ctrl.release(self.cfg["move_key"])

    def _state_wait_play_again(self):
        time.sleep(2)
        self._click_color_button("yellow")
        time.sleep(14)

    def _state_navigate_menu(self):
        self._log("Navigating to main menu")
        ax, ay = self.cfg["home_button"]
        self._click_text("MAIN MENU", timeout=30) or self.ctrl.click_abs(ax, ay)
        time.sleep(1.0)

    def _state_select_brawler(self, slot_key: str):
        ax, ay = self.cfg[slot_key]
        self._log(f"Clicking brawler slot {slot_key} at abs ({ax}, {ay})")
        self.ctrl.click_abs(ax, ay)
        time.sleep(0.5)

    def _state_navigate_mode(self, mode_text: str):
        self._log(f"Navigating to {mode_text}")
        # Open the game mode selection screen
        ox, oy = self.cfg["gamemode_open_button"]
        self.ctrl.click_abs(ox, oy)
        time.sleep(0.8)
        # Duels is off-screen to the right — swipe to bring it into view
        if mode_text.upper() == "DUELS":
            fx, fy = self.cfg["gamemode_swipe_from"]
            tx, ty = self.cfg["gamemode_swipe_to"]
            self._log("Swiping to reveal Duels")
            self.ctrl.swipe(fx, fy, tx, ty)
            time.sleep(0.5)
        if not self._click_text(mode_text, timeout=30):
            raise RuntimeError(f"Could not find '{mode_text}' on screen")
        time.sleep(0.8)
        if not self._click_text("PLAY", timeout=15):
            raise RuntimeError("Could not find PLAY button")
        time.sleep(1.0)

    def _state_brawlball_queue(self):
        self._log("In queue — waiting for game to start")
        # Game has started when we can no longer see any menu/queue text
        deadline = time.time() + 120
        while time.time() < deadline:
            img = self._screenshot()
            in_menu, _ = self.detector.find_text(img, "PLAY")
            if not in_menu:
                self._log("Match started")
                return
            time.sleep(self.poll)
        raise RuntimeError("Timed out waiting for Brawl Ball match to start")

    def _state_brawlball_game(self):
        self._log(f"Brawl Ball game — holding W, spamming {self.cfg['autoaim_key']!r}")
        pixel = self.cfg.get("brawlball_ingame_pixel")
        self.ctrl.hold(self.cfg["move_key"])
        last_aim = time.time()
        while True:
            img = self._screenshot()
            ended = (not self.detector.pixel_is_blue(img, pixel[0], pixel[1])
                     if pixel else self.detector.detect_game_end(img))
            if ended:
                self._log("Brawl Ball game ended")
                break
            now = time.time()
            if now - last_aim >= self.autoaim_interval:
                self.ctrl.press(self.cfg["autoaim_key"])
                last_aim = now
            time.sleep(0.1)
        self.ctrl.release(self.cfg["move_key"])

    def _dismiss_interstitials(self, timeout: float = 15.0):
        """Click through rank-up / trophy road cutscenes blocking navigation."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            img = self._screenshot()
            found, center = self.detector.find_lets_go(img)
            if found and center:
                self._log("Rank-up screen — clicking LET'S GO")
                r = self.capture.rect
                self.ctrl.click_abs(r["left"] + center[0], r["top"] + center[1])
                time.sleep(1.2)
            else:
                break

    # --- main loop ----------------------------------------------------------

    def run(self):
        print("Bot started. Move mouse to top-left to abort (pyautogui failsafe).")
        print(f"Starting state: {self.state.name}")
        self.capture.focus()
        try:
            while True:
                if self.state == State.INITIAL_PLAY:
                    self._state_initial_play()
                    self.state = State.DUELS_GAME_1

                elif self.state == State.DUELS_GAME_1:
                    self._state_duels_game(1)
                    self.state = State.WAIT_PLAY_AGAIN

                elif self.state == State.WAIT_PLAY_AGAIN:
                    self._state_wait_play_again()
                    self.state = State.DUELS_GAME_2

                elif self.state == State.DUELS_GAME_2:
                    self._state_duels_game(2)
                    self.state = State.NAVIGATE_MENU

                elif self.state == State.NAVIGATE_MENU:
                    self._state_navigate_menu()
                    self.state = State.SELECT_BB_BRAWLER

                elif self.state == State.SELECT_BB_BRAWLER:
                    self._state_select_brawler("brawler_slot_brawlball")
                    self.state = State.NAVIGATE_BRAWLBALL

                elif self.state == State.NAVIGATE_BRAWLBALL:
                    self._state_navigate_mode("BRAWL BALL")
                    self.state = State.BRAWLBALL_QUEUE

                elif self.state == State.BRAWLBALL_QUEUE:
                    self._state_brawlball_queue()
                    self.state = State.BRAWLBALL_GAME

                elif self.state == State.BRAWLBALL_GAME:
                    self._state_brawlball_game()
                    for _ in range(3):
                        self._click_color_button("blue", region_frac=(0.5, 0.8, 1.0, 1.0))
                        time.sleep(1.0)
                    self._dismiss_interstitials()
                    self.state = State.NAVIGATE_MENU_2

                elif self.state == State.NAVIGATE_MENU_2:
                    self._state_navigate_menu()
                    self.state = State.RESTORE_BRAWLER

                elif self.state == State.RESTORE_BRAWLER:
                    self._state_select_brawler("brawler_slot_duels")
                    self.state = State.NAVIGATE_DUELS

                elif self.state == State.NAVIGATE_DUELS:
                    self._state_navigate_mode("DUELS")
                    self.state = State.DUELS_GAME_1

        except pyautogui.FailSafeException:
            print("\nFailsafe triggered — bot stopped.")
        except KeyboardInterrupt:
            print("\nInterrupted — bot stopped.")
        except Exception as e:
            print(f"\nBot error: {e}")
        finally:
            self.ctrl.release_all()
            print("All keys released.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Brawl Stars automation bot")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without sending input")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    bot = GameBot(cfg, dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
