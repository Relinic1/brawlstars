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
        self._rect = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

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

    def detect_game_end(self, img: np.ndarray) -> bool:
        for keyword in ("PLAY AGAIN", "VICTORY", "DEFEAT", "TAP TO CONTINUE"):
            found, _ = self.find_text(img, keyword)
            if found:
                return True
        return False

    def detect_in_game(self, img: np.ndarray) -> bool:
        """Heuristic: we're still in-game if none of the end-screen markers appear."""
        return not self.detect_game_end(img)

    def crop_region(self, img: np.ndarray, region: list[float]) -> np.ndarray:
        h, w = img.shape[:2]
        x0 = int(region[0] * w)
        y0 = int(region[1] * h)
        x1 = int(region[2] * w)
        y1 = int(region[3] * h)
        return img[y0:y1, x0:x1], (x0, y0)

    def find_lets_go(self, img: np.ndarray) -> tuple[bool, tuple | None]:
        """Detect the green rank-up 'LET'S GO' button in the bottom-right region."""
        h, w = img.shape[:2]
        ox, oy = int(w * 0.5), int(h * 0.7)
        region = img[oy:h, ox:w]

        # Phase 1: fast green pixel check
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        if mask.sum() < 1000:
            return False, None

        # Phase 2: OCR confirm
        for term in ("LET'S GO", "LETS GO", "GO"):
            found, center = self.find_text(region, term)
            if found and center:
                return True, (center[0] + ox, center[1] + oy)

        # Fallback: strong green signal → click centroid even if OCR missed the text
        if mask.sum() > 5000:
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + ox
                cy = int(M["m01"] / M["m00"]) + oy
                return True, (cx, cy)

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

    # --- states -------------------------------------------------------------

    def _state_initial_play(self):
        self._log("Clicking PLAY to start first Duels game")
        ax, ay = self.cfg["home_button"]
        self.ctrl.click_abs(ax, ay)
        self._log("Waiting for game to load...")
        deadline = time.time() + 60
        while time.time() < deadline:
            img = self._screenshot()
            found, _ = self.detector.find_text(img, "PLAY")
            if not found:
                self._log("Game loaded — starting")
                return
            time.sleep(self.poll)

    def _state_duels_game(self, game_num: int):
        self._log(f"Duels game {game_num} started — holding {self.cfg['move_key']!r}")
        self.ctrl.hold(self.cfg["move_key"])
        while True:
            img = self._screenshot()
            if self.detector.detect_game_end(img):
                self._log("Game ended detected")
                break
            time.sleep(self.poll)
        self.ctrl.release(self.cfg["move_key"])

    def _state_wait_play_again(self):
        self._log("Waiting for PLAY AGAIN...")
        if not self._click_text("PLAY AGAIN"):
            raise RuntimeError("PLAY AGAIN button not found in time")
        time.sleep(1.5)  # wait for next game to load

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
        self.ctrl.hold(self.cfg["move_key"])
        last_aim = time.time()
        while True:
            img = self._screenshot()
            if self.detector.detect_game_end(img):
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
                    self._click_text("PROCEED")
                    self._click_text("PROCEED")
                    self._click_text("EXIT")
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
