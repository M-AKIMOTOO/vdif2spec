use std::fs::File;
use std::io::{BufWriter, Write, Read, Seek, SeekFrom, Error, ErrorKind};
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::Flag;
use gnuplot::{Figure, AxesCommon, Fix};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle}; // indicatif クレートのインポート
use std::path::{Path, PathBuf};


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

    #[arg(long, default_value = "octadisk", help = "Recording type (octadisk or vsrec)")]
    rec: String,
}

fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size <= 1 {
        return data.to_vec();
    }
    data.windows(window_size).map(|window| window.iter().sum::<f32>() / window_size as f32).collect()
}

fn process_vdif_frames(filename: &str, fft_size: usize, skip_time: usize, length: usize, bit_depth: u8, bit_shuffle: bool) -> Result<impl Iterator<Item = Vec<f32>>, Error> {
    
    let file = File::open(filename)?;
    let data_size = 256 * 1024 * 1024; // 1 秒あたり 256 MB

    let skip_byte = data_size * skip_time;
    // スキップ処理
    let mut file = std::io::BufReader::new(file);
    file.seek(SeekFrom::Start(skip_byte as u64))?;

    let data_frames_iter = (0..length).map(move |_| {
        let mut buffer = vec![0u8; data_size];
            match file.read_exact(&mut buffer) {
            Ok(_) => {
                        decode_vdif_data(&buffer, bit_depth, bit_shuffle)
                            .chunks(fft_size)
                            .map(|chunk| chunk.to_vec())
                            .collect::<Vec<_>>()
            }
            Err(_) => vec![], // エラー時は空の Vec を返す
        }
    }).flat_map(|chunks| chunks.into_iter());
    Ok(data_frames_iter)
}

fn bit_shuffle(data: &mut [u8]) {
    for chunk in data.chunks_mut(4) {
        if chunk.len() == 4 {
            let original = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let shuffled = shuffle_bits(original);
            let shuffled_bytes = shuffled.to_le_bytes();
            chunk.copy_from_slice(&shuffled_bytes);
        }
    }
}

fn shuffle_bits(original: u32) -> u32 {
    let mut shuffled = 0;
    shuffled |= (original >> 24) & 0xFF; // 24-31
    shuffled |= ((original >> 16) & 0xFF) << 8; // 16-23
    shuffled |= ((original >> 8) & 0xFF) << 16; // 8-15
    shuffled |= (original & 0xFF) << 24; // 0-7

    let mut result = 0u32;
    //24,25,26,27,28,29,30,31
    result |= ((shuffled >> 24) & 0xFF) << 24; //24-31->24-31
    //16,17,18,19,20,21,22,23
    result |= ((shuffled >> 16) & 0xFF) << 16; //16-23->16-23
    //8,9,10,11,12,13,14,15
    result |= ((shuffled >> 8) & 0xFF) << 8; //8-15->8-15
    //0,1,2,3,4,5,6,7
    result |= (shuffled & 0xFF) << 0; //0-7 ->0-7

    let mut new_result = 0u32;

    new_result |= ((result >> 31) & 0x01) << 24;//31->24
    new_result |= ((result >> 30) & 0x01) << 25;//30->25
    new_result |= ((result >> 29) & 0x01) << 26;//29->26
    new_result |= ((result >> 28) & 0x01) << 27;//28->27
    new_result |= ((result >> 27) & 0x01) << 28;//27->28
    new_result |= ((result >> 26) & 0x01) << 29;//26->29
    new_result |= ((result >> 25) & 0x01) << 30;//25->30
    new_result |= ((result >> 24) & 0x01) << 31;//24->31

    new_result |= ((result >> 23) & 0x01) << 16;//23->16
    new_result |= ((result >> 22) & 0x01) << 17;//22->17
    new_result |= ((result >> 21) & 0x01) << 18;//21->18
    new_result |= ((result >> 20) & 0x01) << 19;//20->19
    new_result |= ((result >> 19) & 0x01) << 20;//19->20
    new_result |= ((result >> 18) & 0x01) << 21;//18->21
    new_result |= ((result >> 17) & 0x01) << 22;//17->22
    new_result |= ((result >> 16) & 0x01) << 23;//16->23

    new_result |= ((result >> 15) & 0x01) << 8;//15->8
    new_result |= ((result >> 14) & 0x01) << 9;//14->9
    new_result |= ((result >> 13) & 0x01) << 10;//13->10
    new_result |= ((result >> 12) & 0x01) << 11;//12->11
    new_result |= ((result >> 11) & 0x01) << 12;//11->12
    new_result |= ((result >> 10) & 0x01) << 13;//10->13
    new_result |= ((result >> 9) & 0x01) << 14;//9->14
    new_result |= ((result >> 8) & 0x01) << 15;//8->15

    new_result |= ((result >> 7) & 0x01) << 0;//7->0
    new_result |= ((result >> 6) & 0x01) << 1;//6->1
    new_result |= ((result >> 5) & 0x01) << 2;//5->2
    new_result |= ((result >> 4) & 0x01) << 3;//4->3
    new_result |= ((result >> 3) & 0x01) << 4;//3->4
    new_result |= ((result >> 2) & 0x01) << 5;//2->5
    new_result |= ((result >> 1) & 0x01) << 6;//1->6
    new_result |= ((result >> 0) & 0x01) << 7;//0->7
    
    new_result
}


fn decode_vdif_data(raw_data: &[u8], bit_depth: u8, rec_vsrec: bool) -> Vec<f32> {

    let mut data = raw_data.to_vec();
    if rec_vsrec {
        bit_shuffle(&mut data);
    }
    let raw_data = &data;
    
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

fn plot_spectrum(freqs: &[f32], spectrum: &[f32], filename: &str, average: usize, xmin: f64, xmax: f64) {

    let smoothed_freqs: Vec<f32> = moving_average(&freqs, average);
    let smoothed_spectrum: Vec<f32> = moving_average(&spectrum, average);

    let mut fg = Figure::new();
    let axes = fg.axes2d()
        .set_x_label("Frequency (MHz)", &[])
        .set_x_range(Fix(xmin), Fix(xmax)) // 0から帯域幅までを指定
        .set_y_label("Power", &[]);

    axes.lines(freqs, spectrum, &[gnuplot::Caption("Spectrum"), gnuplot::LineWidth(2.0)]);
    axes.lines(smoothed_freqs, smoothed_spectrum, &[gnuplot::Caption("Moving Average"), gnuplot::LineWidth(2.0)]);

    fg.save_to_png(filename, 800, 600).expect("Failed to save plot");

    println!("Spectrum data saved to: {}", filename);
}

fn is_power_of_two(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    (n & (n - 1)) == 0
}

fn filepath(path_str: &str) -> PathBuf {
    let path = Path::new(path_str);

    // 拡張子を除いたパスを取得
    let mut base_path = path.to_path_buf();
    if path.extension().is_some() {
        base_path.set_extension("");
    }

    base_path
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

    // rec の値が正しいかチェック
    if args.rec != "octadisk" && args.rec != "vsrec" {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("rec value must be 'octadisk' or 'vsrec', but got {}", args.rec),
        ));
    }

    let rec_vsrec = args.rec == "vsrec";

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
    println!("  Moveing avg : {}", args.avg);
    println!("  Recorder    : {}", args.rec);
    println!("  Bit shuffle : {}", rec_vsrec);
    println!("#--------------------#");
    

    let base_path = filepath(&args.ifile);
    let base_path = base_path.to_string_lossy();


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

    let data_frames = process_vdif_frames(&args.ifile, args.fft, args.skip, args.length, args.bit, rec_vsrec)?;
    for frame in data_frames {
      let spectrum = fft_processor.process(&frame);
      for (i, val) in spectrum.iter().enumerate() {
          integrated_spectrum[i] += val / (args.fft as f32 * args.length as f32);
      }
      pb.inc(1);
    };
    // 周波数軸の生成 (DC 成分を除外)
    let freq_step = args.bw / (args.fft as f32 / 2.0);
    let freqs: Vec<f32> = (1..args.fft / 2).map(|i| i as f32 * freq_step).collect();
    
    // スペクトルプロット
    plot_spectrum(&freqs, &integrated_spectrum, &format!("{}_spec.png", base_path), args.avg, 0.0, args.bw as f64);
    
    if args.fmin != 0.0 || args.fmax != 512.0 {
        plot_spectrum(&freqs, &integrated_spectrum, &format!("{}_peak.png", base_path), args.avg, args.fmin, args.fmax);
    }

    // スペクトルデータのテキスト出力
    if args.output {
        let filename = format!("{}_spec.txt", base_path);
        let file = File::create(&filename)?;
        let mut ofile = BufWriter::new(file);

        writeln!(ofile, "# VDIF FFT Spectrum Analyzer Data")?;
        writeln!(ofile, "# Inputed Parameters")?;
        writeln!(ofile, "# File        : {}", args.ifile)?;
        writeln!(ofile, "# FFT         : {}", args.fft)?;
        writeln!(ofile, "# Skip (sec)  : {}", args.skip)?;
        writeln!(ofile, "# Length (sec): {}", args.length)?;
        writeln!(ofile, "# Bit         : {}", args.bit)?;
        writeln!(ofile, "# BW (MHz)    : {}", args.bw)?;
        writeln!(ofile, "# moveing avg : {}", args.avg)?;
        writeln!(ofile, "# Frequency(MHz) Spectrum")?;
        for (f, s) in freqs.iter().zip(integrated_spectrum.iter()) {
            writeln!(ofile, "{:.6} {:.6}", f, s)?;
        }
        println!("Spectrum data saved to: {}", filename);
    }

    Ok(())
}
