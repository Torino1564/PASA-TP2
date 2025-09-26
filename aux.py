#Archivo para las definiciones defunciones auxiliares de manera tal de mantener la prolijidad de código y de conceptos del resto de los archivos.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp
from scipy.signal import resample_poly
import warnings


#ignorar casos específicos:
warnings.filterwarnings("ignore", category=UserWarning)        # para wavfile
warnings.filterwarnings("ignore", category=RuntimeWarning)   


def plot_signal(signal, fs, *, start=None, end=None, 
                
                title="Señal", xlabel="Tiempo [s]", ylabel="Amplitud", figsize=(18,6)):
    """
    Grafica una señal con opción de seleccionar un rango de tiempo.

    Parámetros
    ----------
    signal : array-like
        Señal de audio.
    fs : int
        Frecuencia de muestreo.
    start : float o None
        Tiempo inicial en segundos (None = inicio de la señal).
    end : float o None
        Tiempo final en segundos (None = fin de la señal).
    """
    n = np.arange(len(signal))
    t = n / fs

    # Si se pasa rango en tiempo, convertir a índices
    if start is not None:
        start_idx = int(start * fs)
    else:
        start_idx = 0

    if end is not None:
        end_idx = int(end * fs)
    else:
        end_idx = len(signal)

    # Recortar señal y tiempo
    signal = signal[start_idx:end_idx]
    t = t[start_idx:end_idx]

    # Graficar
    plt.figure(figsize=figsize)
    plt.plot(t, signal, color="darkorange")
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_psd_log(frequencies, psd, *,
                 title="Densidad espectral de potencia",
                 xlabel="Frecuencia [Hz]",
                 ylabel="Densidad espectral de potencia [V**2/Hz]",
                 figsize=(18,6)):
    """
    Grafica una densidad espectral de potencia (PSD) con escala logarítmica en el eje X.

    Parámetros
    ----------
    frequencies : array-like
        Vector de frecuencias (Hz).
    psd : array-like
        Valores de densidad espectral de potencia.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje X.
    ylabel : str
        Etiqueta del eje Y.
    figsize : tuple
        Tamaño de la figura (ancho, alto).
    """
    plt.figure(figsize=figsize)
    plt.semilogy(frequencies, psd, color="darkorange")  # escala log en eje Y
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()



def get_graphs(sound, fs, record, fs_record, title,
                  level_mode="none",            # "none" | "rms" | "peak" | "psd_band"
                  psd_band=(200, 2000),         # banda de referencia para "psd_band" [Hz]
                  nperseg_welch=2048,
                  nfft_spec=1024, noverlap_spec=None,
                  cmap="magma", fmax_spec=20_000,
                  share_spec_clim=True,
                  psd_ylim=(-120, 0),
                  y_alpha=0.65, y_ls="--"):
    """
    Grafica:
      (0,0) Señales temporales superpuestas (tras alinear niveles si se pide)
      (0,1) PSD (Welch) superpuestas en dB (no dB/Hz), con compensación de nivel opcional
      (1,0) Espectrograma de x
      (1,1) Espectrograma de y

    level_mode:
      - "none": sin igualación
      - "rms": iguala niveles en tiempo por RMS
      - "peak": iguala por pico
      - "psd_band": compensa offset en dB midiendo la diferencia media en la banda psd_band
    """
    eps = 1e-12
    if noverlap_spec is None:
        noverlap_spec = nfft_spec // 2

    # --- Copias float y quitar DC
    x = np.asarray(sound, dtype=float) - np.mean(sound)
    y = np.asarray(record, dtype=float) - np.mean(record)

    # --- Si Fs distintas: remuestrear y a fs (para temporal)
    if fs_record != fs:
        g = np.gcd(int(fs_record), int(fs))
        y = resample_poly(y, int(fs/g), int(fs_record/g))

    # --- Igualación de nivel en tiempo (si se pide)
    if level_mode in ("rms", "peak"):
        def rms(a): return np.sqrt(np.mean(a**2) + eps)
        if level_mode == "rms":
            scale = rms(x) / (rms(y) + eps)
        else:  # peak
            scale = (np.max(np.abs(x)) + eps) / (np.max(np.abs(y)) + eps)
        y = y * scale

    # --- Recorte común para temporal
    L = min(len(x), len(y))
    t = np.arange(L) / fs
    x_t = x[:L]
    y_t = y[:L]

    # --- Welch (densidad) y paso a potencia por bin (dB)
    fx, Pxx = sp.welch(x, fs=fs, window="hamming", nperseg=nperseg_welch)
    fy, Pyy = sp.welch(record, fs=fs_record, window="hamming", nperseg=nperseg_welch)
    dfx = fx[1] - fx[0]
    dfy = fy[1] - fy[0]
    Pxx_dB = 10*np.log10(np.maximum(Pxx*dfx, eps))
    Pyy_dB_raw = 10*np.log10(np.maximum(Pyy*dfy, eps))

    # Interpolar la PSD de y a la malla de fx para comparar
    Pyy_dB = np.interp(fx, fy, Pyy_dB_raw)

    # --- Igualación de nivel en PSD por banda (si se pide)
    if level_mode == "psd_band":
        f_lo, f_hi = psd_band
        band = (fx >= f_lo) & (fx <= f_hi)
        if np.any(band):
            offset = np.median(Pyy_dB[band] - Pxx_dB[band])
            Pyy_dB = Pyy_dB - offset  # compensa ganancia global en dB

    # --- Figura
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=20)

    # (0,0) Temporales superpuestas
    ax[0,0].plot(t, x_t, label="$x_j(n)$", color="#E66100", lw=1.1)
    ax[0,0].plot(t, y_t, label="$y_j(n)$", color="#5D3A9B", lw=1.1, alpha=y_alpha, ls=y_ls)
    ax[0,0].set_title("Comparación de señales temporales", fontsize=16)
    ax[0,0].set_xlabel("Tiempo [s]", fontsize=14)
    ax[0,0].set_ylabel("Amplitud", fontsize=14)
    ax[0,0].grid(alpha=0.3)
    ax[0,0].legend()

    # (0,1) PSD superpuestas (dB)
    ax[0,1].semilogx(fx, Pxx_dB, label="$R_{xx}(f)$ (dB)", color="#E66100", lw=1.8)
    ax[0,1].semilogx(fx, Pyy_dB, label="$R_{yy}(f)$ (dB)", color="#5D3A9B", lw=1.8, ls="--", alpha=0.9)
    ax[0,1].set_title("Espectro de potencia (Welch)", fontsize=16)
    ax[0,1].set_xlabel("Frecuencia [Hz]", fontsize=14)
    ax[0,1].set_ylabel("Potencia [dB]", fontsize=14)
    ax[0,1].grid(which="both", ls="--", alpha=0.4)
    if psd_ylim is not None:
        ax[0,1].set_ylim(*psd_ylim)
    ax[0,1].legend()

    # (1,0) Espectrograma de x
    Sx, fx_s, tx_s, imx = ax[1,0].specgram(x, NFFT=nfft_spec, Fs=fs,
                                            noverlap=noverlap_spec, cmap=cmap)
    ax[1,0].set_title("Espectrograma de $x_j(n)$", fontsize=16)
    ax[1,0].set_xlabel("t [s]", fontsize=14)
    ax[1,0].set_ylabel("f [Hz]", fontsize=14)
    if fmax_spec is not None:
        ax[1,0].set_ylim(0, fmax_spec)
    cbarx = fig.colorbar(imx, ax=ax[1,0]); cbarx.set_label("dB", fontsize=12)

    # (1,1) Espectrograma de y (remuestreada)
    Sy, fy_s, ty_s, imy = ax[1,1].specgram(y, NFFT=nfft_spec, Fs=fs,
                                            noverlap=noverlap_spec, cmap=cmap)
    ax[1,1].set_title("Espectrograma de $y_j(n)$", fontsize=16)
    ax[1,1].set_xlabel("t [s]", fontsize=14)
    ax[1,1].set_ylabel("f [Hz]", fontsize=14)
    if fmax_spec is not None:
        ax[1,1].set_ylim(0, fmax_spec)
    cbary = fig.colorbar(imy, ax=ax[1,1]); cbary.set_label("dB", fontsize=12)

    # misma escala de color entre espectrogramas
    if share_spec_clim:
        Sx_dB = 10*np.log10(np.maximum(Sx, eps))
        Sy_dB = 10*np.log10(np.maximum(Sy, eps))
        vmin = min(Sx_dB.min(), Sy_dB.min())
        vmax = max(Sx_dB.max(), Sy_dB.max())
        imx.set_clim(vmin, vmax)
        imy.set_clim(vmin, vmax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_two_signals(sig1, sig2, fs, 
                     label1='Señal 1', label2='Señal 2',
                     title='Comparación de señales', 
                     xlabel='Tiempo [s]', ylabel='Amplitud', 
                     figsize=(12,5), alpha2=0.7, xlim=None):

    # Recortar al mínimo largo
    n = min(len(sig1), len(sig2))
    sig1 = sig1[:n]
    sig2 = sig2[:n]

    # Eje de tiempo
    t = np.arange(n) / fs  

    plt.figure(figsize=figsize)
    plt.plot(t, sig1, label=label1)
    plt.plot(t, sig2, alpha=alpha2, label=label2)

    # Aplicar límites si se pide
    if xlim is not None:
        plt.xlim(0,xlim)

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()