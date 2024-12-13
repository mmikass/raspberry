import sounddevice as sd
import numpy as np
import os
import threading
import sys
import termios
import tty

# Audio and visualization parameters
INPUT_DEVICE = 0
SAMPLERATE = 48000
BLOCKSIZE = 1024
CHANNELS = 2
EQUALIZER_BARS = 40  # Number of bars/LEDs
MAX_BAR_HEIGHT = 20  # Max bar height in characters
SMOOTHING_FACTOR = 0.8
SMOOTH_STEP = 0.05

# For storing previous bar heights
prev_bar_heights = np.zeros(EQUALIZER_BARS)

# ANSI color codes for terminal visualization
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

def clear_terminal():
    """Clear the terminal for fresh output."""
    os.system('clear' if os.name == 'posix' else 'cls')

def draw_equalizer(fft_magnitudes, smoothing_factor):
    """Display an equalizer in the terminal."""
    global prev_bar_heights

    # Apply logarithmic scaling and normalization
    fft_magnitudes = np.log1p(fft_magnitudes)  # Log scaling
    fft_magnitudes = fft_magnitudes / np.max(fft_magnitudes)  # Normalize
    bar_heights = (fft_magnitudes * MAX_BAR_HEIGHT).astype(int)

    # Apply smoothing
    smoothed_heights = (smoothing_factor * prev_bar_heights +
                        (1 - smoothing_factor) * bar_heights)
    smoothed_heights = smoothed_heights.astype(int)
    prev_bar_heights = smoothed_heights

    # Clear terminal
    clear_terminal()

    # Draw bars from top to bottom
    for height in range(MAX_BAR_HEIGHT, 0, -1):
        line = ""
        for bar_height in smoothed_heights:
            if bar_height >= height:
                # Determine color based on bar height
                if height <= MAX_BAR_HEIGHT * 0.5:  # Low level (green)
                    line += f"{GREEN}█{RESET}"
                elif height <= MAX_BAR_HEIGHT * 0.8:  # Mid level (yellow)
                    line += f"{YELLOW}█{RESET}"
                else:  # High level (red)
                    line += f"{RED}█{RESET}"
            else:
                line += " "  # Space for empty parts
        print(line)

    # Draw a persistent bottom line
    print("█" * EQUALIZER_BARS)

    # Display smoothing factor
    print(f"Smoothing Factor: {smoothing_factor:.2f} (Adjust with + / -)")

def high_pass_filter(audio_data, cutoff=60, samplerate=44100):
    """Remove low frequencies/DC offset."""
    fft_data = np.fft.rfft(audio_data)
    frequencies = np.fft.rfftfreq(len(audio_data), d=1 / samplerate)
    fft_data[frequencies < cutoff] = 0  # Filter frequencies below cutoff
    return np.fft.irfft(fft_data)

def audio_callback(indata, frames, time, status):
    """Process audio data and update equalizer."""
    global SMOOTHING_FACTOR
    if status:
        print(f"Status: {status}")

    # Convert stereo to mono
    audio_data = np.mean(indata, axis=1)

    # High-pass filter
    filtered_audio = high_pass_filter(audio_data)

    # FFT for frequency analysis
    fft_result = np.fft.rfft(filtered_audio)
    fft_magnitudes = np.abs(fft_result)

    # Split FFT magnitudes into frequency bands
    freq_bands = np.linspace(0, len(fft_magnitudes), EQUALIZER_BARS + 1, dtype=int)
    band_magnitudes = [
        np.mean(fft_magnitudes[freq_bands[i]:freq_bands[i + 1]])
        for i in range(EQUALIZER_BARS)
    ]

    # Draw equalizer
    draw_equalizer(np.array(band_magnitudes), SMOOTHING_FACTOR)

def key_listener():
    """Listen for keypresses to adjust smoothing factor."""
    global SMOOTHING_FACTOR
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == '+':
                SMOOTHING_FACTOR = min(SMOOTHING_FACTOR + SMOOTH_STEP, 1.0)
            elif key == '-':
                SMOOTHING_FACTOR = max(SMOOTHING_FACTOR - SMOOTH_STEP, 0.0)
            elif key == 'q':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

try:
    # Start key listener in a separate thread
    threading.Thread(target=key_listener, daemon=True).start()

    # Start audio stream
    with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=audio_callback
    ):
        print("Equalizer visualization. Press + / - to adjust smoothing. Press 'q' to quit.")
        sd.sleep(1000000)
except KeyboardInterrupt:
    print("Visualization stopped.")
except Exception as e:
    print(f"An error occurred: {e}")
