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



def read_vdif_chunks(filename, fft_size, length, bit_depth):
    """
    VDIF データを 1 秒ごとに読み込むジェネレーター
    """
    data_size = 256 * 1024 * 1024  # 1 秒あたり 256 MB
    
    with open(filename, 'rb') as f:
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

def main():
    parser = argparse.ArgumentParser(description='VDIF FFT Spectrum Analyzer')
    parser.add_argument('--ifile' , type=str, required=True, help='Input VDIF file')
    parser.add_argument('--fft'   , type=int, default=4096, help='FFT size (power of 2)')
    parser.add_argument('--length', type=int, default=1   , help='Integration time (seconds)')
    parser.add_argument('--bit'   , type=int, default=2   , help='Bit depth per sample')
    parser.add_argument('--bw'    , type=int, default=512 , help='BandWidth (MHz)')
    parser.add_argument('--cpu'   , type=int, default=4   , help='Number of CPU cores')
        
    args = parser.parse_args()
    ifile = args.ifile
    fft = args.fft
    length = args.length
    bit = args.bit
    bw = args.bw
    
    fft_check = fft
    while True :
        fft_check /= 2
        if fft_check == 1.0 :
            break
        elif 0.0 < fft_check < 1.0 :
            print("Please select a power-of-2 number (e.g. 1024, 8192, 1048576)")
            exit(1)
    
    # スペクトルの積算用配列を確保
    integrated_spectrum = np.zeros(args.fft // 2)
    
    # 並列処理のセットアップ
    with mp.Pool(args.cpu) as pool:
        tasks = read_vdif_chunks(ifile, fft, length, bit)
        for spectrum in pool.imap_unordered(process_fft, tasks):
            integrated_spectrum += spectrum  # メモリ節約しながら加算
    
    # 規格化
    integrated_spectrum /= fft
    integrated_spectrum[0] = 0.0

    bw_freq = np.linspace(0,bw,fft//2)
    
    # txt に保存
    txt_filename = os.path.splitext(ifile)[0] + '_spectrum.txt'
    np.savetxt(txt_filename, np.column_stack([np.round(bw_freq,10), np.round(integrated_spectrum,10)]), delimiter=' ', header='#Frequency,Amplitude', comments='')
    
    # PDF に保存
    pdf_filename = os.path.splitext(ifile)[0] + '_spectrum.pdf'
    plt.figure(figsize=(9,6))
    plt.plot(bw_freq, integrated_spectrum, "-", label='Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power')
    plt.xlim(xmin=0.0)
    plt.ylim(ymin=0.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_filename)
    plt.show()
    plt.close()
    
    print(f'Spectrum saved: {txt_filename}, {pdf_filename}')

if __name__ == '__main__':
    main()

