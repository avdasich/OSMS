
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

A = 2
f = 4
phi = np.pi / 3

t = np.linspace(0, 1, 1000)
y = A * np.cos(2 * np.pi * f * t + phi)

fs, voice = wavfile.read("OSM_lab1.wav")


def sampling(f_s):
    T_s = 1 / f_s
    t_samples = np.arange(f_s) * T_s
    y_samples = A * np.cos(2 * np.pi * f * t_samples + phi)
    return t_samples, y_samples, T_s


def spectrum(y_samples, T_s):
    fft_values = np.fft.rfft(y_samples)
    frequencies = np.fft.rfftfreq(len(y_samples), d=T_s)
    amplitudes = 2 * np.abs(fft_values) / len(y_samples)
    amplitudes[0] /= 2
    if len(y_samples) % 2 == 0:
        amplitudes[-1] /= 2
    return frequencies, amplitudes


def graphs(t_samples, y_samples, frequencies, amplitudes, f_s):
    plt.figure(figsize=(11, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t, y, label="Исходный сигнал")
    plt.scatter(t_samples, y_samples, color="red", label="Отсчеты")
    plt.title(f"Исходный сигнал и отсчеты при {f_s} Гц")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.stem(frequencies, amplitudes)
    plt.title(f"Амплитудный спектр при {f_s} Гц")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, y, label="Исходный сигнал")
    plt.plot(t_samples, y_samples, "o--", label="Соединенные отсчеты")
    plt.title(f"Восстановление сигнала при {f_s} Гц")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"signal_{f_s}.png")


def analyze_voice(fs, voice):
    N = len(voice)
    duration = N / fs
    type_datas = voice.dtype
    if voice.ndim == 1:
        channels = 1
    else:
        channels = voice.shape[1]
    return N, duration, type_datas, channels


def downsample_voice(fs, voice):
    new_voice = voice[::10]
    new_fs = fs // 10
    return new_voice, new_fs


def voice_graph(fs, voice, N, new_fs, new_voice):
    t_voice = np.arange(N) / fs
    t_new_voice = np.arange(len(new_voice)) / new_fs
    plt.figure(figsize=(11, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_voice, voice, color="purple", label=f"Оригинальный ({fs} Гц)")
    plt.title("Исходная запись голоса")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_new_voice, new_voice, color="orange", label=f"Прореженный ({new_fs} Гц)")
    plt.title("После прореживания в 10 раз")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("voice_waveforms.png")


t9, y9, T9 = sampling(9)
t36, y36, T36 = sampling(36)
freq9, amp9 = spectrum(y9, T9)
freq36, amp36 = spectrum(y36, T36)

print(f"Частота сигнала: {f} Гц; период: {1 / f} с")
for rate, samples, period in [(9, y9, T9), (36, y36, T36)]:
    print(f"{rate} Гц: {len(samples)} отсчетов, шаг {period:.4f} с, память {samples.nbytes} байт")

graphs(t9, y9, freq9, amp9, 9)
graphs(t36, y36, freq36, amp36, 36)

N, duration, type_datas, channels = analyze_voice(fs, voice)
print(f"\nГолос: {fs} Гц, {N} отсчетов, {duration:.4f} с, {type_datas}, каналов: {channels}")

new_voice, new_fs = downsample_voice(fs, voice)
print(f"После прореживания: {new_fs} Гц, {len(new_voice)} отсчетов, {len(new_voice)/new_fs:.4f} с")
wavfile.write("new_OSM_lab1.wav", new_fs, new_voice)
voice_graph(fs, voice, N, new_fs, new_voice)

#до и после прореживания
freq_voice, amp_voice = spectrum(voice, 1 / fs)
freq_new, amp_new = spectrum(new_voice, 1 / new_fs)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freq_voice, amp_voice)
plt.title("Спектр исходного голоса")
plt.xlim(0, 6000)
plt.ylabel("Амплитуда")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(freq_new, amp_new)
plt.title(f"Спектр после прореживания (предел Найквиста: {new_fs / 2:.0f} Гц)")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.tight_layout()
plt.savefig("voice_spectra.png")


def quantization(samples, bits):
    levels = 2 ** bits
    step = 2 * A / (levels - 1)

    indexes = (samples + A) / step
    indexes = np.round(indexes)
    indexes = np.clip(indexes, 0, levels - 1)

    quantized = -A + indexes * step

    return quantized

bits_list = [3, 4, 5, 6]
quantized_signals = []

for bits in bits_list:
    y_quant = quantization(y9, bits)
    error = np.mean(np.abs(y9 - y_quant))

    quantized_signals.append(y_quant)
    print(f"{bits} бит: средняя ошибка = {error:.5f}")


plt.figure(figsize=(10, 5))
plt.plot(freq9, amp9, "k--", label="Исходный спектр")

for i in range(len(bits_list)):
    freq_q, amp_q = spectrum(quantized_signals[i], T9)

    plt.plot(freq_q, amp_q, "o-", label=f"{bits_list[i]} бит")

plt.title("Влияние квантования на спектр")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

