import sounddevice as sd
import numpy as np
import os
import threading
import sys
import termios
import tty
import neopixel
import board

# Audio parameters
INPUT_DEVICE = 1  # Replace with Bluetooth device index
SAMPLE_RATE = 48000
BLOCKSIZE = 1024
CHANNELS = 2
EQUALIZER_BARS = 40
MAX_BAR_HEIGHT = 20
SMOOTHING_FACTOR = 0.8
SMOOTH_STEP = 0.05

# NeoPixel setup
LED_COUNT = EQUALIZER_BARS
LED_PIN = board.D18
BRIGHTNESS = 0.5
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False)

prev_bar_heights = np.zeros(EQUALIZER_BARS)

def clear_terminal():
    os.system('clear' if os.name == 'posix' else 'cls')

def draw_equalizer(fft_magnitudes, smoothing_factor):
    global prev_bar_heights

    fft_magnitudes = np.log1p(fft_magnitudes)
    fft_magnitudes /= np.max(fft_magnitudes)
    bar_heights = (fft_magnitudes * MAX_BAR_HEIGHT).astype(int)

    smoothed_heights = (smoothing_factor * prev_bar_heights +
                        (1 - smoothing_factor) * bar_heights)
    smoothed_heights = smoothed_heights.astype(int)
    prev_bar_heights = smoothed_heights

    clear_terminal()

    for height in range(MAX_BAR_HEIGHT, 0, -1):
        line = ""
        for bar_height in smoothed_heights:
            if bar_height >= height:
                line += "█"
            else:
                line += " "
        print(line)

    for i, height in enumerate(smoothed_heights):
        intensity = int((height / MAX_BAR_HEIGHT) * 255)
        pixels[i] = (intensity, 0, 255 - intensity)
    pixels.show()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Stream status: {status}")
    audio_data = np.mean(indata, axis=1)
    fft_result = np.fft.rfft(audio_data)
    fft_magnitudes = np.abs(fft_result)
    freq_bands = np.linspace(0, len(fft_magnitudes), EQUALIZER_BARS + 1, dtype=int)
    band_magnitudes = [
        np.mean(fft_magnitudes[freq_bands[i]:freq_bands[i + 1]])
        for i in range(EQUALIZER_BARS)
    ]
    draw_equalizer(np.array(band_magnitudes), SMOOTHING_FACTOR)

try:
    with sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=audio_callback
    ):
        print("Visualizing audio input...")
        sd.sleep(1000000)
except KeyboardInterrupt:
    print("Visualization stopped.")
    pixels.fill((0, 0, 0))
    pixels.show()
