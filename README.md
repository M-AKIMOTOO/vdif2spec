Recommended RAM = 32 GB or more

# Rust Version
cargo run --release

[package]
name = "vdif2spec"  
version = "0.1.0"  
edition = "2021"  
rust-version = "1.85"  
authors = ["Masanori AKIMOTO"]  
description = "This program make a spectrum graph of a VDIF format. cargo version: 1.85.0 (d73d2caf9 2024-12-31)，rustc version: 1.85.0 (4d91de4e4 2025-02-17)，rustup version : 1.27.1 (54dd3d00f 2024-04-24)"  


[dependencies]
fftw = { version = "0.8", features = ["system"] }  
gnuplot = "0.0.37"  
clap = { version = "4.4", features = ["derive"] }  
indicatif = "0.17"  
find_peaks = "0.1.5"  

# Python Version
Python 3.10  

import argparse  
import numpy as np  
import scipy.fft  
import multiprocessing as mp  
import matplotlib.pyplot as plt  
import os, sys   
