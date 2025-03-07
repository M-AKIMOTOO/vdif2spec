use std::fs::File;
use std::io::{BufWriter, Write, Read, Seek, SeekFrom, Error, ErrorKind};
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::Flag;
use gnuplot::{Figure, AxesCommon, Color, LineStyle, Dash, Fix};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle}; // indicatif クレートのインポート

#[derive(Parser)]
#[command(
    name = "VDIF FFT Spectrum Analyzer",
    about = "Analyzes VDIF data and generates FFT spectrum.",
    long_about = "This tool processes VDIF (VLBI Data Interchange Format) data, performs FFT (Fast Fourier Transform) analysis, and generates a spectrum plot. It supports specifying input file, FFT size, skip time, processing length, bit depth, bandwidth, output file options and moving average window size",
)]
struct Cli {
    #[arg(long, help = "Input VDIF file name")]
    ifile: String,

    #[arg(long, default_value = "4096", help = "FFT size (must be a power of 2)")]
    fft: usize,

    #[arg(long, default_value = "0", help = "Skip time in seconds")]
    skip: usize,

    #[arg(long, default_value = "1", help = "Processing length in seconds")]
    length: usize,

    #[arg(long, default_value = "2", help = "Bit depth of the VDIF data")]
    bit: u8,

    #[arg(long, default_value = "512", help = "Bandwidth in MHz")]
    bw: f32,

    #[arg(long, help = "Output spectrum data to text file")]
    output: bool,
    
    #[arg(long, default_value = "32", help = "Moving average window size")]
    avg: usize,

    #[arg(long, default_value = "0.0", help = "Minimum of frequency")]
    fmin: f64,
    
    #[arg(long, default_value = "512.0", help = "Maximum of frequency")]
    fmax: f64,
}

fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 {
        return data.to_vec();
    }
    data.windows(window_size).map(|window| window.iter().sum::<f32>() / window_size as f32).collect()
}

fn convole(data: &[f32], window: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];

    for i in 0..data.len() {
        for j in 0..window.len() {
            if i + j < data.len() {
                result[i+j] += data[i] * window[j];
            }
        }
    }
    result
}

fn process_vdif_frames(filename: &str, fft_size: usize, skip_time: usize, length: usize, bit_depth: u8) -> Result<impl Iterator<Item = Vec<f32>>, Error> {
    let mut file = File::open(filename)?;
    let data_size = 256 * 1024 * 1024; // 1 秒あたり 256 MB

    let skip_byte = data_size * skip_time;
    // スキップ処理
    file.seek(SeekFrom::Start(skip_byte as u64))?;

    let data_frames_iter = (0..length).flat_map(move |_| {
        let mut buffer = vec![0u8; data_size];
        match file.read_exact(&mut buffer) {
            Ok(_) => {
                let decoded_data = decode_vdif_data(&buffer, bit_depth, fft_size);
                let chunked_data = decoded_data.chunks(fft_size).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();
                chunked_data.into_iter().map(|x|x).collect::<Vec<_>>()
            }
            Err(_) => vec![], // エラー時は空の Vec を返す
        }
    }).filter(|x| !x.is_empty());
    Ok(data_frames_iter)
}

fn decode_vdif_data(raw_data: &[u8], bit_depth: u8, fft_size: usize) -> Vec<f32> {
    let mask = (1 << bit_depth) - 1; // 例: 2-bit -> 0b11 (3)
    let samples_per_byte = (8 / bit_depth) as usize; // 例: 2-bit -> 4 samples/byte
    let total_samples = raw_data.len() * samples_per_byte;
    
    let mut decoded = vec![0.0; total_samples];

    for (i, &byte) in raw_data.iter().enumerate() {
        for j in 0..samples_per_byte {
            let shift = (bit_depth as usize) * j;
            let sample = ((byte >> shift) & mask) as i8 - (mask / 2) as i8;
            decoded[i * samples_per_byte + j] = sample as f32; // Python の処理を忠実に再現
        }
    }
    decoded
}

struct FFTProcessor {
    plan: R2CPlan32,
    input: AlignedVec<f32>,
    output: AlignedVec<fftw::types::c32>,
}

impl FFTProcessor {
    fn new(fft_size: usize) -> Self {
        let input = AlignedVec::<f32>::new(fft_size);
        let output = AlignedVec::<fftw::types::c32>::new(fft_size / 2 + 1);
        let plan = R2CPlan::aligned(&[fft_size], Flag::MEASURE).unwrap();
        FFTProcessor { plan, input, output }
    }

    fn process(&mut self, data_chunk: &[f32]) -> Vec<f32> {
        self.input.clone_from_slice(data_chunk);
        self.plan.r2c(&mut self.input, &mut self.output).unwrap();
        
        // DC 成分 (index 0) を除外し、それ以降のデータのみを返す
        self.output.iter().skip(1).map(|c| c.norm()).collect()
    }
}

fn plot_spectrum(freqs: &[f32], spectrum: &[f32], filename: &str, smooth_spectrum: Option<&[f32]>, xmin: f64, xmax: f64) {
    let mut fg = Figure::new();
    let axes = fg.axes2d()
        .set_x_label("Frequency (MHz)", &[])
        .set_x_range(Fix(xmin), Fix(xmax)) // 0から帯域幅までを指定
        .set_y_label("Power", &[]);

    axes.lines(freqs, spectrum, &[gnuplot::Caption("Spectrum"), gnuplot::LineWidth(2.0)]);
    if let Some(smoothed) = smooth_spectrum {
        axes.lines(&freqs[..smoothed.len()], smoothed, &[gnuplot::Caption("Moving Average"), gnuplot::LineWidth(2.0)]);
    }

    fg.save_to_png(filename, 800, 600).expect("Failed to save plot");

    println!("Spectrum data saved to: {}", filename);
}

fn is_power_of_two(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    (n & (n - 1)) == 0
}

fn main() -> Result<(), Error> {
    let args = Cli::parse();

    // FFT サイズが 2 のべき乗かどうかをチェック
    if !is_power_of_two(args.fft) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("FFT size must be a power of 2, but got {}", args.fft),
        ));
    }

    // skip が length より大きくないかチェック
    if args.skip > args.length {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("Skip value ({}) cannot be greater than length ({})", args.skip, args.length),
        ));
    }
    

    println!("#--------------------#");
    println!("# Inputed Parameters #");
    println!("#                    #");
    println!("  File        : {}", args.ifile);
    println!("  FFT         : {}", args.fft);
    println!("  Skip (sec)  : {}", args.skip);
    println!("  Length (sec): {}", args.length);
    println!("  Bit         : {}", args.bit);
    println!("  BW (MHz)    : {}", args.bw);
    println!("  Output txt  : {}", args.output);
    println!("  moveing avg : {}", args.avg);
    println!("#--------------------#");
    
    // プログレスバーの初期化
    let pb = ProgressBar::new(args.length as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let mut integrated_spectrum = vec![0.0f32; args.fft / 2 ];

    let mut fft_processor = FFTProcessor::new(args.fft);

    let processed_spectra = process_vdif_frames(&args.ifile, args.fft, args.skip, args.length, args.bit)?;
    processed_spectra.map(|chunk| fft_processor.process(&chunk)).for_each(|spectrum|{
        for (i, val) in spectrum.iter().enumerate() {
            integrated_spectrum[i] += val / (args.fft as f32 * args.length as f32);
        }
        pb.inc(1);
    });

    // 移動平均の計算
    let smoothed_spectrum: Vec<f32> = moving_average(&integrated_spectrum, args.avg);

    // 周波数軸の生成 (DC 成分を除外)
    let freq_step = args.bw / (args.fft as f32 / 2.0);
    let freqs: Vec<f32> = (1..args.fft / 2).map(|i| i as f32 * freq_step).collect();
    
    // スペクトルプロット
    plot_spectrum(&freqs, &integrated_spectrum, &format!("{}_spec.png", args.ifile), Some(&smoothed_spectrum), 0.0, args.bw as f64);
    
    if args.fmin != 0.0 || args.fmax != 512.0 {
        plot_spectrum(&freqs, &integrated_spectrum, &format!("{}_peak.png", args.ifile), Some(&smoothed_spectrum), args.fmin, args.fmax);
    }

    // スペクトルデータのテキスト出力
    if args.output {
        let filename = format!("{}_spec.txt", args.ifile);
        let file = File::create(&filename)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# Frequency(MHz) Spectrum")?;
        for (f, s) in freqs.iter().zip(integrated_spectrum.iter()) {
            writeln!(writer, "{:.6} {:.6}", f, s)?;
        }
        println!("Spectrum data saved to: {}", filename);
    }

    // プログレスバーを終了
    // /pb.finish_with_message("Processing complete");

    /* 
    // 標準偏差を計算し、5σ以上のデータを抽出
    let std_dev = (integrated_spectrum.iter().map(|&x| x * x).sum::<f32>() / (integrated_spectrum.len() as f32) - integrated_spectrum.iter().sum::<f32>().powi(2) / (integrated_spectrum.len() as f32).powi(2)).sqrt();
    let mean = integrated_spectrum.iter().sum::<f32>() / integrated_spectrum.len() as f32;
    let threshold = 5.0 * std_dev;

    let significant_indices: Vec<usize> = smoothed_spectrum.iter().enumerate().filter(|(i, &s)| s > threshold+mean).map(|(i, _)| i).collect();

    if !significant_indices.is_empty() {
        //ピークを抽出
        let mut significant_spectrum: Vec<f32> = Vec::new();
        let mut significant_spectrum_smooth: Vec<f32> = Vec::new();

        for index in significant_indices.iter(){
            significant_spectrum.push(integrated_spectrum[*index]);
            significant_spectrum_smooth.push(smoothed_spectrum[*index]);
        }
        let significant_indices_avg = if significant_indices.is_empty() {
            0.0
        } else {
            significant_indices.iter().map(|&x| x as f32).sum::<f32>() / significant_indices.len() as f32
        };

        //ピークがある周波数を特定
        let significant_freq_index = significant_indices_avg.round() as usize;
        let center_freq = freqs[significant_freq_index];

        // 周波数帯域を絞る
        let start_freq = (center_freq - 3.0);
        let end_freq = (center_freq + 3.0);

        let start_index = freqs.iter().position(|&f| f >= start_freq).unwrap_or(0);
        let end_index = freqs.iter().position(|&f| f >= end_freq).unwrap_or(freqs.len());

        // ピーク付近をプロット
        plot_spectrum(&freqs[start_index..end_index], &integrated_spectrum[start_index..end_index], &format!("{}_peak.png", args.ifile),Some(&smoothed_spectrum[start_index..end_index]));
    }
    */


    Ok(())
}
