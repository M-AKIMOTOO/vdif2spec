use clap::{Parser, ValueEnum};
use image::ImageReader;
use plotters::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Error as IoError, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_LEVELS_2BIT: [f32; 4] = [-1.5, 0.5, -0.5, 1.5];
const DEFAULT_SHUFFLE_EXTERNAL: [usize; 32] = [
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
    8, 7, 6, 5, 4, 3, 2, 1, 0,
];
const VSREC_LEVELS_2BIT: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
const VSREC_SHUFFLE_EXTERNAL: [usize; 32] = [
    24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14,
    15, 0, 1, 2, 3, 4, 5, 6, 7,
];
const OUTPUT_AUTO_SENTINEL: &str = "__AUTO__";

type DynError = Box<dyn Error + Send + Sync>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Sideband {
    #[value(name = "USB")]
    Usb,
    #[value(name = "LSB")]
    Lsb,
}

impl fmt::Display for Sideband {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sideband::Usb => write!(f, "USB"),
            Sideband::Lsb => write!(f, "LSB"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "vdif2spec",
    about = "Create VDIF/raw spectrum using phased_array/vdif_chN_spliter-compatible decode",
    arg_required_else_help = true
)]
struct Cli {
    #[arg(long, value_name = "FILE", help = "Input VDIF/raw file path")]
    vdif: PathBuf,

    #[arg(long, default_value_t = 16384, value_name = "N", help = "FFT points per frame (power-of-two)")]
    fft: usize,

    #[arg(long, default_value_t = 0.0, value_name = "SEC", help = "Skip time [s]")]
    skip: f64,

    #[arg(long, value_name = "SEC", help = "Processing length [s]")]
    length: Option<f64>,

    #[arg(long = "loop", default_value_t = 0, value_name = "N", help = "Max FFT frame count to integrate (0 = no limit)")]
    loop_count: usize,

    #[arg(long, default_value_t = 2, value_name = "BITS", help = "Bit depth [bits/sample]")]
    bit: usize,

    #[arg(long = "level", allow_negative_numbers = true, num_args = 1.., value_name = "LEVEL", help = "Quantization levels in code order (count must be 2^bit)")]
    level: Vec<f32>,

    #[arg(long = "shuffle", num_args = 1.., value_name = "bit31...bit00", help = "Bit shuffle map (32 entries, permutation of 0..31)")]
    shuffle: Vec<usize>,

    #[arg(long, help = "Force VSREC decode map (--shuffle/--level override)")]
    vsrec: bool,

    #[arg(long = "bw", default_value_t = 1024.0, value_name = "MHz", help = "Sampling bandwidth [MHz]. Spectrum span is 0..bw/2")]
    bw: f64,

    #[arg(long = "if", allow_hyphen_values = true, num_args = 2, value_names = ["LOW", "HIGH"], help = "IF range [MHz] to extract inside 0..bw/2")]
    if_range: Option<Vec<f64>>,

    #[arg(long, value_enum, default_value_t = Sideband::Lsb, help = "Input sideband convention")]
    sideband: Sideband,

    #[arg(long, default_value_t = 1, value_name = "N", help = "CPU worker threads")]
    cpu: usize,

    #[arg(long, default_value_t = 2000, value_name = "FRAMES", help = "Progress print interval [frames]")]
    progress_every: u64,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "FILE",
        num_args = 0..=1,
        default_missing_value = OUTPUT_AUTO_SENTINEL,
        help = "Save spectrum text. With no value: same dir/stem as plot (.txt)"
    )]
    output: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShuffleKind {
    Identity,
    PairSwap,
    Generic,
}

struct DecodePlan {
    bits: usize,
    shuffle_kind: ShuffleKind,
    input_shifts: [u32; 32],
}

struct FftWorker {
    forward: Arc<dyn RealToComplex<f32>>,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
}

impl FftWorker {
    fn new(fft: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft);
        let spec = forward.make_output_vec();
        Self {
            forward,
            time: vec![0.0; fft],
            spec,
        }
    }
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn default_levels(bit: usize) -> Vec<f32> {
    let n = 1usize << bit;
    let center = (n as f32 - 1.0) / 2.0;
    (0..n).map(|i| i as f32 - center).collect()
}

fn resolve_levels(bit: usize, values: &[f32]) -> Result<Vec<f32>, DynError> {
    if bit == 0 {
        return Err("--bit must be >= 1".into());
    }
    let expected = 1usize << bit;
    if values.is_empty() {
        if bit == 2 {
            return Ok(DEFAULT_LEVELS_2BIT.to_vec());
        }
        return Ok(default_levels(bit));
    }
    let levels = values.to_vec();
    if levels.len() != expected {
        return Err(format!(
            "expected {expected} levels for {bit}-bit quantization, got {}",
            levels.len()
        )
        .into());
    }
    Ok(levels)
}

fn resolve_shuffle(values: &[usize]) -> Result<Vec<usize>, DynError> {
    let values_external = if values.is_empty() {
        DEFAULT_SHUFFLE_EXTERNAL.to_vec()
    } else {
        values.to_vec()
    };
    if values_external.len() != 32 {
        return Err("shuffle map must contain exactly 32 entries".into());
    }

    let mut sorted = values_external.clone();
    sorted.sort_unstable();
    for (expected, &found) in (0usize..32).zip(sorted.iter()) {
        if expected != found {
            return Err("shuffle map must be a permutation of 0..31".into());
        }
    }

    // Convert external order (MSB->LSB) to internal bit index order.
    let mut values_internal = vec![0usize; 32];
    for (idx_msb_to_lsb, input_bit) in values_external.into_iter().enumerate() {
        let out_bit_lsb = 31 - idx_msb_to_lsb;
        values_internal[out_bit_lsb] = input_bit;
    }
    Ok(values_internal)
}

fn build_decode_plan(bits: usize, shuffle_in: &[usize]) -> Result<DecodePlan, DynError> {
    if shuffle_in.len() != 32 {
        return Err("internal shuffle map must have 32 entries".into());
    }

    let mut input_shifts = [0u32; 32];
    for (idx, &mapped) in shuffle_in.iter().enumerate() {
        input_shifts[idx] = mapped as u32;
    }

    let is_identity = shuffle_in.iter().enumerate().all(|(i, &v)| i == v);
    let is_pair_swap = shuffle_in.iter().enumerate().all(|(i, &v)| (i ^ 1) == v);
    let shuffle_kind = if is_identity {
        ShuffleKind::Identity
    } else if is_pair_swap {
        ShuffleKind::PairSwap
    } else {
        ShuffleKind::Generic
    };

    Ok(DecodePlan {
        bits,
        shuffle_kind,
        input_shifts,
    })
}

fn decode_block_into_with_plan(
    raw: &[u8],
    levels: &[f32],
    samples: usize,
    plan: &DecodePlan,
    output: &mut [f32],
    lsb_to_usb: bool,
) -> Result<(), DynError> {
    if output.len() < samples {
        return Err("decode output buffer is too small".into());
    }

    let bits = plan.bits;
    let code_mask = (1u64 << bits) - 1;

    let mut out_idx = 0usize;
    let mut acc = 0u64;
    let mut acc_bits = 0usize;

    for chunk in raw.chunks_exact(4) {
        let mut word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

        if plan.shuffle_kind == ShuffleKind::PairSwap {
            word = ((word & 0xAAAA_AAAA) >> 1) | ((word & 0x5555_5555) << 1);
        } else if plan.shuffle_kind == ShuffleKind::Generic {
            let mut shuffled = 0u32;
            for (out_bit, &in_shift) in plan.input_shifts.iter().enumerate() {
                shuffled |= ((word >> in_shift) & 1) << out_bit;
            }
            word = shuffled;
        }

        acc |= (word as u64) << acc_bits;
        acc_bits += 32;

        while acc_bits >= bits && out_idx < samples {
            let code = (acc & code_mask) as usize;
            let mut value = levels[code];
            if lsb_to_usb && (out_idx & 1) == 1 {
                value = -value;
            }
            output[out_idx] = value;
            out_idx += 1;
            acc >>= bits;
            acc_bits -= bits;
        }
    }

    if out_idx != samples {
        return Err(format!("decoded {} samples, expected {}", out_idx, samples).into());
    }

    Ok(())
}

fn resolve_plot_path(vdif: &Path) -> PathBuf {
    let parent = vdif
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let out_dir = parent.join("vdif2spec");
    let stem = vdif
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("spectrum");
    out_dir.join(format!("{stem}_vdif2spec_spec.png"))
}

fn resolve_text_path(vdif: &Path, out_opt: &Option<String>) -> Option<PathBuf> {
    match out_opt {
        None => None,
        Some(s) if s == OUTPUT_AUTO_SENTINEL => {
            let mut p = resolve_plot_path(vdif);
            p.set_extension("txt");
            Some(p)
        }
        Some(s) => Some(PathBuf::from(s)),
    }
}

fn plot_spectrum(
    freqs_mhz: &[f64],
    power: &[f32],
    out_png: &Path,
    if_range_mhz: (f64, f64),
) -> Result<(), DynError> {
    if freqs_mhz.is_empty() || power.is_empty() || freqs_mhz.len() != power.len() {
        return Err("no spectrum points to plot".into());
    }

    if let Some(parent) = out_png.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let y_min_raw = power
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::INFINITY, f64::min)
        .min(0.0);
    let y_max_raw = power
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let (y_min, y_max) = if y_max_raw > y_min_raw {
        let pad = (y_max_raw - y_min_raw) * 0.05;
        (y_min_raw - pad, y_max_raw + pad)
    } else {
        (y_min_raw - 1.0, y_min_raw + 1.0)
    };

    let font_scale = 1.3_f64;
    let caption_size = 28_i32;
    let axis_desc_size = (24.0 * font_scale).round() as i32;
    let tick_label_size = (18.0 * font_scale).round() as i32;
    let x_label_area = (64.0 * font_scale).round() as u32;
    let y_label_area = (80.0 * font_scale).round() as u32;

    let root = BitMapBackend::new(out_png, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(24)
        .caption("VDIF Spectrum (a.u.)", ("sans-serif", caption_size))
        .x_label_area_size(x_label_area)
        .y_label_area_size(y_label_area)
        .build_cartesian_2d(if_range_mhz.0..if_range_mhz.1, y_min..y_max)?;

    let yfmt = |y: &f64| format!("{y:.1e}");
    chart
        .configure_mesh()
        .x_desc("IF Frequency (MHz)")
        .y_desc("Power (a.u.)")
        .axis_desc_style(("sans-serif", axis_desc_size))
        .label_style(("sans-serif", tick_label_size))
        .y_label_formatter(&yfmt)
        .light_line_style(TRANSPARENT)
        .draw()?;

    chart.draw_series(LineSeries::new(
        freqs_mhz
            .iter()
            .zip(power.iter())
            .map(|(&f, &p)| (f, p as f64)),
        &BLUE,
    ))?;

    root.present()?;
    quantize_png_in_place(out_png)?;
    Ok(())
}


fn quantize_png_in_place(path: &Path) -> Result<(), DynError> {
    let dyn_image = ImageReader::open(path)?.decode()?;
    let rgba = dyn_image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let rgba_pixels: Vec<imagequant::RGBA> = rgba
        .as_raw()
        .chunks_exact(4)
        .map(|px| imagequant::RGBA::new(px[0], px[1], px[2], px[3]))
        .collect();

    let mut liq = imagequant::new();
    liq.set_max_colors(16)?;
    liq.set_quality(0, 35)?;
    liq.set_speed(1)?;

    let mut img = liq.new_image(rgba_pixels, width as usize, height as usize, 0.0)?;
    let mut res = liq.quantize(&mut img)?;
    res.set_dithering_level(0.0)?;

    let (palette, pixels) = res.remapped(&mut img)?;

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Best);

    let mut pal_rgb = Vec::with_capacity(palette.len() * 3);
    let mut pal_alpha = Vec::with_capacity(palette.len());
    let mut has_alpha = false;
    for c in palette {
        pal_rgb.push(c.r);
        pal_rgb.push(c.g);
        pal_rgb.push(c.b);
        pal_alpha.push(c.a);
        if c.a < 255 {
            has_alpha = true;
        }
    }
    encoder.set_palette(pal_rgb);
    if has_alpha {
        encoder.set_trns(pal_alpha);
    }

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(&pixels)?;
    Ok(())
}

fn save_spectrum_text(path: &Path, freqs_mhz: &[f64], power: &[f32], cli: &Cli, if_range: (f64, f64), frames: u64) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# vdif2spec spectrum")?;
    writeln!(writer, "# vdif      : {}", cli.vdif.display())?;
    writeln!(writer, "# fft       : {}", cli.fft)?;
    writeln!(writer, "# bit       : {}", cli.bit)?;
    writeln!(writer, "# bw        : {:.6} MHz", cli.bw)?;
    writeln!(writer, "# sideband  : {}", cli.sideband)?;
    writeln!(writer, "# skip      : {:.6} s", cli.skip)?;
    if let Some(length) = cli.length {
        writeln!(writer, "# length    : {:.6} s", length)?;
    }
    writeln!(writer, "# loop      : {}", cli.loop_count)?;
    writeln!(writer, "# integrated frames: {}", frames)?;
    writeln!(writer, "# IF range  : {:.6} .. {:.6} MHz", if_range.0, if_range.1)?;
    writeln!(writer, "# Frequency(MHz) Power(a.u.)")?;
    for (&f, &p) in freqs_mhz.iter().zip(power.iter()) {
        writeln!(writer, "{f:.9} {p:.9e}")?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), IoError> {
    let mut cli = Cli::parse();
    if cli.vsrec {
        if cli.bit != 2 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "--vsrec requires --bit 2",
            ));
        }
        cli.level = VSREC_LEVELS_2BIT.to_vec();
        cli.shuffle = VSREC_SHUFFLE_EXTERNAL.to_vec();
    }

    if !is_power_of_two(cli.fft) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("--fft must be power-of-two, got {}", cli.fft),
        ));
    }
    if cli.bit == 0 {
        return Err(IoError::new(ErrorKind::InvalidInput, "--bit must be >= 1"));
    }
    if cli.bw <= 0.0 {
        return Err(IoError::new(ErrorKind::InvalidInput, "--bw must be > 0"));
    }
    if cli.skip < 0.0 {
        return Err(IoError::new(ErrorKind::InvalidInput, "--skip must be >= 0"));
    }
    if let Some(length) = cli.length {
        if length <= 0.0 {
            return Err(IoError::new(ErrorKind::InvalidInput, "--length must be > 0"));
        }
    }
    if cli.cpu == 0 {
        return Err(IoError::new(ErrorKind::InvalidInput, "--cpu must be >= 1"));
    }
    if cli.progress_every == 0 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "--progress-every must be >= 1",
        ));
    }

    let bw_half_mhz = cli.bw / 2.0;
    let if_range_mhz = if let Some(raw) = cli.if_range.as_ref() {
        let (low, high) = (raw[0], raw[1]);
        if !low.is_finite() || !high.is_finite() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "--if values must be finite",
            ));
        }
        if low >= high {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "--if low must be less than high",
            ));
        }
        if low < 0.0 || high > bw_half_mhz {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                format!(
                    "--if must be within 0..{:.6} MHz, got {:.6}..{:.6}",
                    bw_half_mhz, low, high
                ),
            ));
        }
        (low, high)
    } else {
        (0.0, bw_half_mhz)
    };

    let levels = resolve_levels(cli.bit, &cli.level)
        .map_err(|e| IoError::new(ErrorKind::InvalidInput, e.to_string()))?;
    let shuffle = resolve_shuffle(&cli.shuffle)
        .map_err(|e| IoError::new(ErrorKind::InvalidInput, e.to_string()))?;
    let shuffle_external_log = if cli.shuffle.is_empty() {
        DEFAULT_SHUFFLE_EXTERNAL.to_vec()
    } else {
        cli.shuffle.clone()
    };
    let decode_plan = build_decode_plan(cli.bit, &shuffle)
        .map_err(|e| IoError::new(ErrorKind::InvalidInput, e.to_string()))?;

    let frame_bits = cli.fft * cli.bit;
    if frame_bits % 32 != 0 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("fft*bit (= {frame_bits}) must be 32-bit word aligned"),
        ));
    }
    let frame_bytes = frame_bits / 8;

    let sampling_hz = cli.bw * 1e6;
    let bytes_per_sec = sampling_hz * cli.bit as f64 / 8.0;

    let total_bytes = std::fs::metadata(&cli.vdif)?.len();
    let skip_bytes_raw = (cli.skip * bytes_per_sec).floor() as u64;
    let skip_bytes = (skip_bytes_raw / frame_bytes as u64) * frame_bytes as u64;
    if skip_bytes >= total_bytes {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "--skip exceeds input length",
        ));
    }

    let available_frames = (total_bytes - skip_bytes) / frame_bytes as u64;
    let length_frames = if let Some(length) = cli.length {
        ((length * sampling_hz / cli.fft as f64).floor() as u64).max(1)
    } else {
        available_frames
    };
    let loop_frames = if cli.loop_count == 0 {
        u64::MAX
    } else {
        cli.loop_count as u64
    };
    let process_frames = available_frames.min(length_frames).min(loop_frames);
    if process_frames == 0 {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "nothing to process (0 frames)",
        ));
    }

    let half_bins = cli.fft / 2;
    let start_bin = ((if_range_mhz.0 / bw_half_mhz) * half_bins as f64).floor() as usize;
    let mut end_bin = ((if_range_mhz.1 / bw_half_mhz) * half_bins as f64).ceil() as usize;
    if end_bin > half_bins {
        end_bin = half_bins;
    }
    if start_bin >= end_bin {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "selected --if range maps to zero bins",
        ));
    }

    let selected_low = start_bin as f64 * bw_half_mhz / half_bins as f64;
    let selected_high = end_bin as f64 * bw_half_mhz / half_bins as f64;

    println!("# vdif2spec");
    println!("  vdif             : {}", cli.vdif.display());
    println!("  fft              : {}", cli.fft);
    println!("  bit              : {}", cli.bit);
    println!("  vsrec            : {}", cli.vsrec);
    println!(
        "  level            : {}",
        levels
            .iter()
            .map(|v| format!("{v:+.6}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  shuffle          : {}",
        shuffle_external_log
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("  bw               : {:.6} MHz", cli.bw);
    println!("  span             : 0 .. {:.6} MHz", bw_half_mhz);
    println!("  if               : {:.6} .. {:.6} MHz", selected_low, selected_high);
    println!("  sideband         : {}", cli.sideband);
    println!("  skip             : {:.6} s", cli.skip);
    if let Some(length) = cli.length {
        println!("  length           : {:.6} s", length);
    }
    println!("  loop             : {}", cli.loop_count);
    println!("  cpu              : {}", cli.cpu);
    println!("  process frames   : {}", process_frames);

    let mut reader = BufReader::new(File::open(&cli.vdif)?);
    reader.seek(SeekFrom::Start(skip_bytes))?;

    let pool = ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build()
        .map_err(|e| IoError::new(ErrorKind::Other, format!("failed to configure rayon: {e}")))?;

    let chunk_frames = (cli.progress_every as usize).max(cli.cpu * 16).max(1);
    let mut in_chunk = vec![0u8; chunk_frames * frame_bytes];
    let input_lsb = cli.sideband == Sideband::Lsb;

    let mut integrated = vec![0.0f32; half_bins];
    let mut done_frames = 0u64;

    while done_frames < process_frames {
        let this_chunk = ((process_frames - done_frames) as usize).min(chunk_frames);
        let in_bytes = this_chunk * frame_bytes;
        reader.read_exact(&mut in_chunk[..in_bytes])?;

        let chunk_accum = pool
            .install(|| -> Result<Vec<f32>, DynError> {
                in_chunk[..in_bytes]
                    .par_chunks(frame_bytes)
                    .try_fold(
                        || (FftWorker::new(cli.fft), vec![0.0f32; half_bins]),
                        |(mut worker, mut acc), raw_frame| {
                            decode_block_into_with_plan(
                                raw_frame,
                                &levels,
                                cli.fft,
                                &decode_plan,
                                &mut worker.time,
                                input_lsb,
                            )?;
                            worker.forward.process(&mut worker.time, &mut worker.spec)?;
                            for (dst, src) in acc.iter_mut().zip(worker.spec.iter()) {
                                *dst += src.norm_sqr();
                            }
                            Ok((worker, acc))
                        },
                    )
                    .map(|res| res.map(|(_, acc)| acc))
                    .try_reduce(
                        || vec![0.0f32; half_bins],
                        |mut a, b| {
                            for (dst, src) in a.iter_mut().zip(b.iter()) {
                                *dst += *src;
                            }
                            Ok(a)
                        },
                    )
            })
            .map_err(|e| IoError::new(ErrorKind::Other, e.to_string()))?;

        for (dst, src) in integrated.iter_mut().zip(chunk_accum.iter()) {
            *dst += *src;
        }

        done_frames += this_chunk as u64;
        if done_frames % cli.progress_every == 0 || done_frames == process_frames {
            let pct = (done_frames as f64 / process_frames as f64) * 100.0;
            print!("\r  progress         : {done_frames}/{process_frames} frames ({pct:.2}%)");
            std::io::stdout().flush()?;
            if done_frames == process_frames {
                println!();
            }
        }
    }

    let mut norm = integrated.iter().map(|&v| v as f64).sum::<f64>();
    norm *= process_frames as f64;
    let inv = if norm > 0.0 { (1.0 / norm) as f32 } else { 1.0f32 };
    for v in &mut integrated {
        *v *= inv;
    }

    let mut freqs_mhz = Vec::with_capacity(end_bin - start_bin);
    let mut power = Vec::with_capacity(end_bin - start_bin);
    for k in start_bin..end_bin {
        let freq = k as f64 * bw_half_mhz / half_bins as f64;
        freqs_mhz.push(freq);
        power.push(integrated[k]);
    }

    let plot_path = resolve_plot_path(&cli.vdif);
    plot_spectrum(&freqs_mhz, &power, &plot_path, (selected_low, selected_high))
        .map_err(|e| IoError::new(ErrorKind::Other, e.to_string()))?;
    println!("  plot             : {}", plot_path.display());

    if let Some(txt_path) = resolve_text_path(&cli.vdif, &cli.output) {
        save_spectrum_text(
            &txt_path,
            &freqs_mhz,
            &power,
            &cli,
            (selected_low, selected_high),
            process_frames,
        )
        .map_err(|e| IoError::new(ErrorKind::Other, e.to_string()))?;
        println!("  output txt       : {}", txt_path.display());
    }

    println!("  done             : {}", cli.vdif.display());
    Ok(())
}
