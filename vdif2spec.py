#!/usr/bin/env python3
# AKIMOTO & ChatGPT
# 2025/03/01

import argparse
import numpy as np
import scipy.fft
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys 

#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"]     = "in"       
plt.rcParams["ytick.direction"]     = "in"       
plt.rcParams["xtick.minor.visible"] = True       
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"]           = True
plt.rcParams["xtick.bottom"]        = True
plt.rcParams["ytick.left"]          = True
plt.rcParams["ytick.right"]         = True  
plt.rcParams["xtick.major.size"]    = 5          
plt.rcParams["ytick.major.size"]    = 5          
plt.rcParams["xtick.minor.size"]    = 3          
plt.rcParams["ytick.minor.size"]    = 3          
plt.rcParams["axes.grid"]           = False
plt.rcParams["grid.color"]          = "lightgray"
plt.rcParams["axes.labelsize"]      = 15
plt.rcParams["font.size"]           = 12


def read_vdif_chunks(filename, fft_size, skip_time, length, bit_depth):
    """
    VDIF データをスキップ時間分スキップし、指定時間分読み込むジェネレーター
    """
    data_size = 256 * 1024 * 1024  # 1 秒あたり 256 MB
    
    with open(filename, 'rb') as f:
        # 指定した時間分スキップ
        f.seek(data_size * skip_time, os.SEEK_SET)
        
        for _ in range(length):
            raw_data = np.frombuffer(f.read(data_size), dtype=np.uint8)
            if len(raw_data) < data_size:
                break
            yield from decode_vdif_data(raw_data, bit_depth, fft_size)

def decode_vdif_data(raw_data, bit_depth, fft_size):
    """
    VDIF のビット深度に応じてデータを復号し、FFT サイズごとに小分け
    """
    mask = (1 << bit_depth) - 1  # 例: 2-bit -> 0b11 (3)
    samples_per_byte = 8 // bit_depth  # 例: 2-bit -> 4 samples/byte
    
    for i in range(0, len(raw_data), fft_size // samples_per_byte):
        chunk = raw_data[i:i + fft_size // samples_per_byte]
        decoded = np.zeros(len(chunk) * samples_per_byte, dtype=np.int8)
        
        for j in range(samples_per_byte):
            shift = bit_depth * j
            decoded[j::samples_per_byte] = ((chunk >> shift) & mask) - (mask // 2)

        yield decoded

def process_fft(data_chunk):
    """
    FFT を計算し、DC 成分を 0 にして振幅を規格化する
    """
    fft_size = len(data_chunk)
    spectrum = np.abs(scipy.fft.fft(data_chunk))[:fft_size // 2]
    return spectrum

def moving_average(data, window_size):
    """
    移動平均を計算する
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    parser = argparse.ArgumentParser(description='VDIF FFT Spectrum Analyzer')
    parser.add_argument('--ifile' , type=str, required=True, help='Input VDIF file')
    parser.add_argument('--fft'   , type=int, default=4096, help='FFT size (power of 2)')
    parser.add_argument('--skip'  , type=int, default=0   , help='Skip time (seconds)')
    parser.add_argument('--length', type=int, default=1   , help='Integration time (seconds)')
    parser.add_argument('--bit'   , type=int, default=2   , help='Bit depth per sample')
    parser.add_argument('--bw'    , type=int, default=512 , help='BandWidth (MHz)')
    parser.add_argument('--cpu'   , type=int, default=4   , help='Number of CPU cores')
    parser.add_argument('--output', action="store_true"   , help='Output spectrum data')
    parser.add_argument('--avg'   , type=int, default=5   , help='Moving average window size')
        
    args = parser.parse_args()
    ifile = args.ifile
    fft = args.fft
    skip = args.skip
    length = args.length
    bit = args.bit
    bw = args.bw
    output = args.output
    avg_window = args.avg
    
    print("#--------------------#")
    print("# Inputed Parameters #")
    print("#                    #")
    print(f"  File        : {ifile}")
    print(f"  FFT         : {fft}")
    print(f"  Skip (sec)  : {skip}")
    print(f"  Length (sec): {length}")
    print(f"  Bit         : {bit}")
    print(f"  BW (MHz)    : {bw}")
    print(f"  Output txt  : {output}")
    print(f"  moveing avg : {avg_window}")
    print("#--------------------#")
    
    # スペクトルの積算用配列を確保
    integrated_spectrum = np.zeros(args.fft // 2)
    
    # 並列処理のセットアップ
    with mp.Pool(args.cpu) as pool:
        tasks = read_vdif_chunks(ifile, fft, skip, length, bit)
        for spectrum in pool.imap_unordered(process_fft, tasks):
            integrated_spectrum += spectrum  # メモリ節約しながら加算
    
    # 規格化
    integrated_spectrum /= fft
    integrated_spectrum *= length
    integrated_spectrum[0] = 0.0
    
    bw_freq = np.linspace(0, bw, fft//2)
    
    # 移動平均の計算
    smoothed_spectrum = moving_average(integrated_spectrum, avg_window)
    
    bw_freq_smooth = bw_freq[:len(smoothed_spectrum)]
    
    # PDF に保存
    pdf_filename = os.path.splitext(ifile)[0] + '_spec.pdf'
    plt.figure(figsize=(9,6))
    plt.plot(bw_freq[1:], integrated_spectrum[1:], "-", label='Spectrum')
    plt.plot(bw_freq_smooth[1:], smoothed_spectrum[1:], "--", label='Moving Average')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power')
    plt.xlim(xmin=0.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_filename)
    #plt.show()
    plt.close()
    
    print(f'Spectrum saved: {pdf_filename}')
    
    
    # 標準偏差を計算し、5σ以上のデータを抽出
    std_dev = np.std(smoothed_spectrum)
    mean    = np.mean(smoothed_spectrum)
    threshold = 5 * std_dev

    significant_indices = np.where(smoothed_spectrum > threshold+mean)[0]
    if len(significant_indices) != 0 :

        significant_indices = int(np.mean(significant_indices))
        
        # 高強度成分を別の Figure にプロット
        center_freq = bw_freq[significant_indices]
        freq_range = (bw_freq >= center_freq - 3) & (bw_freq <= center_freq + 3)
        smoothed_center_freq = bw_freq_smooth[significant_indices]
        smoothed_freq_range = (bw_freq_smooth >= smoothed_center_freq - 3) & (bw_freq_smooth <= smoothed_center_freq + 3)

        pdf_filename = os.path.splitext(ifile)[0] + '_peak.pdf'
        plt.figure(figsize=(9,6))
        plt.plot(bw_freq[freq_range], integrated_spectrum[freq_range], "-", c="tab:red", label='Spectrum')
        plt.plot(bw_freq_smooth[smoothed_freq_range], smoothed_spectrum[smoothed_freq_range], "--", c="tab:green", label='Moving Average')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power')
        plt.legend()
        plt.tight_layout()
        plt.savefig(pdf_filename)
        #plt.show()
        plt.close()

        print(f'Spectrum saved: {pdf_filename}')
    else :
        pass

if __name__ == '__main__':
    main()

