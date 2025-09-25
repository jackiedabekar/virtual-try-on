import pyautogui
import time
import random
import warnings

# Suppress SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set pyautogui to fail-safe mode
pyautogui.FAILSAFE = True


def move_mouse():
    screen_width, screen_height = pyautogui.size()
    x = random.randint(100, screen_width - 100)
    y = random.randint(100, screen_height - 100)
    pyautogui.moveTo(x, y, duration=0.5)


# Main loop to keep the laptop active for 5 hours
def keep_active():
    print(
        "Starting keep-active script for 5 hours. Move mouse to upper-left corner to stop."
    )
    start_time = time.time()
    duration = 5 * 3600  # 5 hours in seconds
    click_alternate = False  # To alternate clicks
    cycle = 0

    try:
        while time.time() - start_time < duration:
            cycle += 1
            elapsed_hours = (time.time() - start_time) / 3600
            print(f"Starting cycle {cycle}. Time elapsed: {elapsed_hours:.2f} hours")

            try:
                move_mouse()

                # Simulate slow scroll down (100 small scrolls with delay for smoothness)
                print("Scrolling down slowly...")
                for _ in range(100):
                    pyautogui.scroll(-10)  # Negative for down
                    time.sleep(0.05)  # Short delay for slow scrolling effect

                # Simulate slow scroll up
                print("Scrolling up slowly...")
                for _ in range(100):
                    pyautogui.scroll(10)  # Positive for up
                    time.sleep(0.05)

                # Alternate mouse clicks to simulate usage (left click every other time)
                if click_alternate:
                    pyautogui.click()  # Left click
                click_alternate = not click_alternate

            except Exception as e:
                print(f"Error in cycle {cycle}: {e}")

            # Calculate remaining time and sleep accordingly (adjusting for time spent in activity ~10 seconds)
            remaining = duration - (time.time() - start_time)
            sleep_time = min(50, remaining)
            if sleep_time > 0:
                print(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

    except pyautogui.FailSafeException:
        print("Script stopped by fail-safe (mouse moved to upper-left corner).")
    except KeyboardInterrupt:
        print("Script stopped by user.")
    finally:
        print("Script completed or stopped. Screen activity simulation ended.")


if __name__ == "__main__":
    keep_active()


# xhost +local:
# python3 auto_run.py
