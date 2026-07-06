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
use std::io::{
    BufRead, BufReader, BufWriter, Error as IoError, ErrorKind, Read, Seek, SeekFrom, Write,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_LEVELS_2BIT: [f32; 4] = [-1.5, 0.5, -0.5, 1.5];
const DEFAULT_SHUFFLE_EXTERNAL: [usize; 32] = [
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0,
];
const VSREC_LEVELS_2BIT: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
const VSREC_SHUFFLE_EXTERNAL: [usize; 32] = [
    24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7,
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

    #[arg(
        long,
        default_value_t = 16384,
        value_name = "N",
        help = "FFT points per frame (power-of-two)"
    )]
    fft: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "SEC",
        help = "Skip time [s]"
    )]
    skip: f64,

    #[arg(long, value_name = "SEC", help = "Processing length [s]")]
    length: Option<f64>,

    #[arg(
        long = "loop",
        default_value_t = 0,
        value_name = "N",
        help = "Number of output segments. Each segment spans --length seconds (0 = single output)"
    )]
    loop_count: usize,

    #[arg(
        long,
        default_value_t = 2,
        value_name = "BITS",
        help = "Bit depth [bits/sample]"
    )]
    bit: usize,

    #[arg(long = "level", allow_negative_numbers = true, num_args = 1.., value_name = "LEVEL", help = "Quantization levels in code order (count must be 2^bit)")]
    level: Vec<f32>,

    #[arg(long = "shuffle", num_args = 1.., value_name = "bit31...bit00", help = "Bit shuffle map (32 entries, permutation of 0..31)")]
    shuffle: Vec<usize>,

    #[arg(long, help = "Force VSREC decode map (--shuffle/--level override)")]
    vsrec: bool,

    #[arg(
        long = "bw",
        default_value_t = 1024.0,
        value_name = "MHz",
        help = "Sampling bandwidth [MHz]. Spectrum span is 0..bw/2"
    )]
    bw: f64,

    #[arg(long = "if", allow_hyphen_values = true, num_args = 2, value_names = ["LOW", "HIGH"], help = "IF range [MHz] to extract inside 0..bw/2")]
    if_range: Option<Vec<f64>>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Bandpass table (frequency[MHz] value); calibrate by channel-wise division"
    )]
    bptable: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Sideband::Lsb, help = "Input sideband convention")]
    sideband: Sideband,

    #[arg(
        long,
        default_value_t = 1,
        value_name = "N",
        help = "CPU worker threads"
    )]
    cpu: usize,

    #[arg(
        long,
        default_value_t = 2000,
        value_name = "FRAMES",
        help = "Progress print interval [frames]"
    )]
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
    byte_lut: Option<ByteDecodeLut>,
}

struct ByteDecodeLut {
    samples_per_byte: usize,
    table_plain: Box<[[f32; 8]; 256]>,
    table_even: Box<[[f32; 8]; 256]>,
    table_odd: Box<[[f32; 8]; 256]>,
}

struct FftWorker {
    forward: Arc<dyn RealToComplex<f32>>,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
    accum: Vec<f32>,
}

impl FftWorker {
    fn new(forward: Arc<dyn RealToComplex<f32>>, half_bins: usize) -> Self {
        let fft = forward.len();
        let spec = forward.make_output_vec();
        Self {
            forward,
            time: vec![0.0; fft],
            spec,
            accum: vec![0.0; half_bins],
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

fn build_decode_plan(
    bits: usize,
    shuffle_in: &[usize],
    levels: &[f32],
) -> Result<DecodePlan, DynError> {
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

    let byte_lut = build_byte_decode_lut(bits, shuffle_kind, levels);

    Ok(DecodePlan {
        bits,
        shuffle_kind,
        input_shifts,
        byte_lut,
    })
}

fn build_byte_decode_lut(
    bits: usize,
    shuffle_kind: ShuffleKind,
    levels: &[f32],
) -> Option<ByteDecodeLut> {
    if bits == 0 || bits > 8 || (8 % bits) != 0 {
        return None;
    }

    let samples_per_byte = 8 / bits;
    let code_mask = (1u16 << bits) - 1;
    let mut table_plain = [[0.0f32; 8]; 256];
    let mut table_even = [[0.0f32; 8]; 256];
    let mut table_odd = [[0.0f32; 8]; 256];

    for byte in 0u16..=255 {
        let source = if shuffle_kind == ShuffleKind::PairSwap {
            ((byte & 0xAA) >> 1) | ((byte & 0x55) << 1)
        } else {
            byte
        };

        for sample_idx in 0..samples_per_byte {
            let shift = sample_idx * bits;
            let code = ((source >> shift) & code_mask) as usize;
            let value = levels[code];
            table_plain[byte as usize][sample_idx] = value;
            table_even[byte as usize][sample_idx] =
                if (sample_idx & 1) == 1 { -value } else { value };
            table_odd[byte as usize][sample_idx] =
                if (sample_idx & 1) == 0 { -value } else { value };
        }
    }

    Some(ByteDecodeLut {
        samples_per_byte,
        table_plain: Box::new(table_plain),
        table_even: Box::new(table_even),
        table_odd: Box::new(table_odd),
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

    if let Some(lut) = plan.byte_lut.as_ref() {
        return decode_block_with_byte_lut(raw, samples, plan, output, lsb_to_usb, lut);
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

fn decode_block_with_byte_lut(
    raw: &[u8],
    samples: usize,
    plan: &DecodePlan,
    output: &mut [f32],
    lsb_to_usb: bool,
    lut: &ByteDecodeLut,
) -> Result<(), DynError> {
    let mut out_idx = 0usize;

    for chunk in raw.chunks_exact(4) {
        let bytes = if plan.shuffle_kind == ShuffleKind::Generic {
            let mut shuffled = 0u32;
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            for (out_bit, &in_shift) in plan.input_shifts.iter().enumerate() {
                shuffled |= ((word >> in_shift) & 1) << out_bit;
            }
            shuffled.to_le_bytes()
        } else {
            [chunk[0], chunk[1], chunk[2], chunk[3]]
        };

        for &byte in &bytes {
            let table = if !lsb_to_usb {
                &lut.table_plain
            } else if (out_idx & 1) == 1 {
                &lut.table_odd
            } else {
                &lut.table_even
            };
            let decoded = &table[byte as usize];
            for &value in decoded[..lut.samples_per_byte].iter() {
                if out_idx >= samples {
                    break;
                }
                output[out_idx] = value;
                out_idx += 1;
            }
        }
    }

    if out_idx != samples {
        return Err(format!("decoded {} samples, expected {}", out_idx, samples).into());
    }

    Ok(())
}

fn resolve_plot_path(vdif: &Path, segment_idx: usize, segment_total: u64) -> PathBuf {
    let parent = vdif
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let out_dir = parent.join("vdif2spec");
    let stem = vdif
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("spectrum");
    if segment_total <= 1 {
        out_dir.join(format!("{stem}_vdif2spec_spec.png"))
    } else {
        out_dir.join(format!(
            "{stem}_vdif2spec_spec_loop{:04}.png",
            segment_idx + 1
        ))
    }
}

fn with_loop_suffix(path: &Path, segment_idx: usize) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let mut out = path.to_path_buf();
    let ext = path.extension().and_then(|e| e.to_str());
    match ext {
        Some(ext) if !ext.is_empty() => {
            out.set_file_name(format!("{stem}_loop{:04}.{ext}", segment_idx + 1))
        }
        _ => out.set_file_name(format!("{stem}_loop{:04}", segment_idx + 1)),
    }
    out
}

fn resolve_text_path(
    vdif: &Path,
    out_opt: &Option<String>,
    segment_idx: usize,
    segment_total: u64,
) -> Option<PathBuf> {
    match out_opt {
        None => None,
        Some(s) if s == OUTPUT_AUTO_SENTINEL => {
            let mut p = resolve_plot_path(vdif, segment_idx, segment_total);
            p.set_extension("txt");
            Some(p)
        }
        Some(s) => {
            let p = PathBuf::from(s);
            if segment_total <= 1 {
                Some(p)
            } else {
                Some(with_loop_suffix(&p, segment_idx))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Spectrum {
    freqs_mhz: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RebinDirection {
    None,
    BandpassToData,
    DataToBandpass,
}

impl fmt::Display for RebinDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RebinDirection::None => write!(f, "none"),
            RebinDirection::BandpassToData => write!(f, "bandpass -> data grid"),
            RebinDirection::DataToBandpass => write!(f, "data -> bandpass grid"),
        }
    }
}

fn read_spectrum_table(path: &Path) -> Result<Spectrum, DynError> {
    let reader = BufReader::new(File::open(path)?);
    parse_spectrum_table(reader, &path.display().to_string())
}

fn parse_spectrum_table<R: BufRead>(reader: R, source: &str) -> Result<Spectrum, DynError> {
    let mut freqs_mhz = Vec::new();
    let mut values = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        let data = line.split_once('#').map_or(line.as_str(), |(head, _)| head);
        let mut columns = data.split_whitespace();
        let Some(freq_raw) = columns.next() else {
            continue;
        };
        let value_raw = columns
            .next()
            .ok_or_else(|| format!("{source}:{}: expected frequency and value", line_idx + 1))?;
        let freq = freq_raw.parse::<f32>().map_err(|e| {
            format!(
                "{source}:{}: invalid frequency '{freq_raw}': {e}",
                line_idx + 1
            )
        })?;
        let value = value_raw.parse::<f32>().map_err(|e| {
            format!(
                "{source}:{}: invalid value '{value_raw}': {e}",
                line_idx + 1
            )
        })?;
        if !freq.is_finite() || !value.is_finite() {
            return Err(format!("{source}:{}: values must be finite", line_idx + 1).into());
        }
        if freqs_mhz.last().is_some_and(|&previous| freq <= previous) {
            return Err(format!(
                "{source}:{}: frequencies must be strictly increasing",
                line_idx + 1
            )
            .into());
        }
        freqs_mhz.push(freq);
        values.push(value);
    }

    if freqs_mhz.len() < 2 {
        return Err(format!("{source}: at least two spectrum rows are required").into());
    }

    Ok(Spectrum { freqs_mhz, values })
}

fn bin_edges(freqs_mhz: &[f32]) -> Result<Vec<f32>, DynError> {
    if freqs_mhz.len() < 2 {
        return Err("at least two frequency channels are required".into());
    }

    let mut edges = Vec::with_capacity(freqs_mhz.len() + 1);
    edges.push(freqs_mhz[0] - (freqs_mhz[1] - freqs_mhz[0]) * 0.5);
    for pair in freqs_mhz.windows(2) {
        if pair[1] <= pair[0] {
            return Err("frequency channels must be strictly increasing".into());
        }
        edges.push((pair[0] + pair[1]) * 0.5);
    }
    let last = freqs_mhz.len() - 1;
    edges.push(freqs_mhz[last] + (freqs_mhz[last] - freqs_mhz[last - 1]) * 0.5);
    Ok(edges)
}

fn mean_channel_width(freqs_mhz: &[f32]) -> f32 {
    (freqs_mhz[freqs_mhz.len() - 1] - freqs_mhz[0]) / (freqs_mhz.len() - 1) as f32
}

fn same_frequency_grid(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() || a.len() < 2 {
        return false;
    }
    let channel_tolerance = mean_channel_width(a).abs().min(mean_channel_width(b).abs()) * 1.0e-3;
    a.iter().zip(b).all(|(&left, &right)| {
        let rounding_tolerance = f32::EPSILON * left.abs().max(right.abs()).max(1.0) * 4.0;
        (left - right).abs() <= channel_tolerance.max(rounding_tolerance)
    })
}

fn rebin_onto(source: &Spectrum, target_freqs_mhz: &[f32]) -> Result<Vec<Option<f32>>, DynError> {
    let source_edges = bin_edges(&source.freqs_mhz)?;
    let target_edges = bin_edges(target_freqs_mhz)?;
    let mut rebinned = Vec::with_capacity(target_freqs_mhz.len());
    let mut source_idx = 0usize;

    for target_idx in 0..target_freqs_mhz.len() {
        let low = target_edges[target_idx];
        let high = target_edges[target_idx + 1];
        let target_width = high - low;
        while source_idx < source.values.len() && source_edges[source_idx + 1] <= low {
            source_idx += 1;
        }

        let mut idx = source_idx;
        let mut weighted_sum = 0.0f32;
        let mut covered_width = 0.0f32;
        while idx < source.values.len() && source_edges[idx] < high {
            let overlap = high.min(source_edges[idx + 1]) - low.max(source_edges[idx]);
            if overlap > 0.0 {
                weighted_sum += source.values[idx] * overlap;
                covered_width += overlap;
            }
            idx += 1;
        }

        let fully_covered = covered_width + target_width * 1.0e-4 >= target_width;
        if fully_covered && covered_width > 0.0 {
            rebinned.push(Some(weighted_sum / covered_width));
        } else {
            rebinned.push(None);
        }
    }

    Ok(rebinned)
}

fn calibrate_bandpass(
    data: &Spectrum,
    bandpass: &Spectrum,
) -> Result<(Spectrum, RebinDirection), DynError> {
    if same_frequency_grid(&data.freqs_mhz, &bandpass.freqs_mhz) {
        let values = data
            .values
            .iter()
            .zip(&bandpass.values)
            .enumerate()
            .map(|(idx, (&value, &bp))| {
                if bp <= 0.0 {
                    Err(format!("bandpass channel {idx} must be greater than zero").into())
                } else {
                    Ok(value / bp)
                }
            })
            .collect::<Result<Vec<_>, DynError>>()?;
        return Ok((
            Spectrum {
                freqs_mhz: data.freqs_mhz.clone(),
                values,
            },
            RebinDirection::None,
        ));
    }

    let data_is_coarser =
        mean_channel_width(&data.freqs_mhz) >= mean_channel_width(&bandpass.freqs_mhz);
    let (target, rebinned, direction, data_on_target) = if data_is_coarser {
        (
            data,
            rebin_onto(bandpass, &data.freqs_mhz)?,
            RebinDirection::BandpassToData,
            true,
        )
    } else {
        (
            bandpass,
            rebin_onto(data, &bandpass.freqs_mhz)?,
            RebinDirection::DataToBandpass,
            false,
        )
    };

    let mut freqs_mhz = Vec::with_capacity(target.freqs_mhz.len());
    let mut values = Vec::with_capacity(target.values.len());
    for (idx, rebinned_value) in rebinned.into_iter().enumerate() {
        let Some(rebinned_value) = rebinned_value else {
            continue;
        };
        let (data_value, bp_value) = if data_on_target {
            (target.values[idx], rebinned_value)
        } else {
            (rebinned_value, target.values[idx])
        };
        if bp_value <= 0.0 {
            return Err(format!(
                "bandpass at {:.9} MHz must be greater than zero",
                target.freqs_mhz[idx]
            )
            .into());
        }
        freqs_mhz.push(target.freqs_mhz[idx]);
        values.push(data_value / bp_value);
    }

    if freqs_mhz.len() < 2 {
        return Err(
            "data and bandpass tables have fewer than two fully overlapping channels".into(),
        );
    }

    Ok((Spectrum { freqs_mhz, values }, direction))
}

fn plot_spectrum(
    freqs_mhz: &[f32],
    power: &[f32],
    out_png: &Path,
    if_range_mhz: (f32, f32),
) -> Result<(), DynError> {
    if freqs_mhz.is_empty() || power.is_empty() || freqs_mhz.len() != power.len() {
        return Err("no spectrum points to plot".into());
    }

    if let Some(parent) = out_png.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let y_min_raw = power.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
    let y_max_raw = power.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (y_min, y_max) = if y_max_raw > y_min_raw {
        let pad = (y_max_raw - y_min_raw) * 0.05;
        (y_min_raw - pad, y_max_raw + pad)
    } else {
        (y_min_raw - 1.0, y_min_raw + 1.0)
    };

    let font_scale = 1.3_f32;
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

    let yfmt = |y: &f32| format!("{y:.1e}");
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
        freqs_mhz.iter().zip(power.iter()).map(|(&f, &p)| (f, p)),
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

fn save_spectrum_text(
    path: &Path,
    freqs_mhz: &[f32],
    power: &[f32],
    cli: &Cli,
    if_range: (f32, f32),
    frames: u64,
    calibration_rebin: Option<RebinDirection>,
) -> Result<(), DynError> {
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
    if let Some(path) = cli.bptable.as_deref() {
        writeln!(writer, "# bptable   : {}", path.display())?;
        writeln!(
            writer,
            "# calibration: power / bandpass ({})",
            calibration_rebin.unwrap_or(RebinDirection::None)
        )?;
    }
    writeln!(writer, "# loop      : {}", cli.loop_count)?;
    writeln!(writer, "# integrated frames: {}", frames)?;
    writeln!(
        writer,
        "# IF range  : {:.6} .. {:.6} MHz",
        if_range.0, if_range.1
    )?;
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
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "--length must be > 0",
            ));
        }
    }
    if cli.loop_count > 0 && cli.length.is_none() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "--loop requires --length",
        ));
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
    let decode_plan = build_decode_plan(cli.bit, &shuffle, &levels)
        .map_err(|e| IoError::new(ErrorKind::InvalidInput, e.to_string()))?;
    let bandpass = cli
        .bptable
        .as_deref()
        .map(read_spectrum_table)
        .transpose()
        .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;

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
    let segment_frames = if cli.loop_count == 0 {
        length_frames.min(available_frames)
    } else {
        length_frames
    };
    if segment_frames == 0 {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "nothing to process (0 frames per segment)",
        ));
    }
    let segment_total = if cli.loop_count == 0 {
        1u64
    } else {
        (cli.loop_count as u64).min(available_frames / segment_frames)
    };
    if segment_total == 0 {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "requested loop segments exceed available input length",
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
    println!(
        "  if               : {:.6} .. {:.6} MHz",
        selected_low, selected_high
    );
    println!("  sideband         : {}", cli.sideband);
    if let Some(path) = cli.bptable.as_deref() {
        println!("  bptable          : {}", path.display());
        println!(
            "  bptable channels : {}",
            bandpass.as_ref().map_or(0, |table| table.values.len())
        );
    }
    println!("  skip             : {:.6} s", cli.skip);
    if let Some(length) = cli.length {
        println!("  length           : {:.6} s", length);
    }
    println!("  loop             : {}", cli.loop_count);
    println!("  cpu              : {}", cli.cpu);
    println!("  frames/segment   : {}", segment_frames);
    println!("  segments         : {}", segment_total);

    let mut reader = BufReader::new(File::open(&cli.vdif)?);
    reader.seek(SeekFrom::Start(skip_bytes))?;

    let pool = ThreadPoolBuilder::new()
        .num_threads(cli.cpu)
        .build()
        .map_err(|e| IoError::other(format!("failed to configure rayon: {e}")))?;
    let mut planner = RealFftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(cli.fft);

    let chunk_frames = (cli.progress_every as usize).max(cli.cpu * 16).max(1);
    let mut in_chunk = vec![0u8; chunk_frames * frame_bytes];
    let input_lsb = cli.sideband == Sideband::Lsb;
    let freqs_mhz: Vec<f32> = (start_bin..end_bin)
        .map(|k| (k as f32) * (bw_half_mhz as f32) / (half_bins as f32))
        .collect();
    let if_range_selected = (selected_low as f32, selected_high as f32);

    for seg_idx in 0..segment_total as usize {
        let mut integrated = vec![0.0f32; half_bins];
        let mut done_frames = 0u64;

        while done_frames < segment_frames {
            let this_chunk = ((segment_frames - done_frames) as usize).min(chunk_frames);
            let in_bytes = this_chunk * frame_bytes;
            reader.read_exact(&mut in_chunk[..in_bytes])?;

            let chunk_accum = pool
                .install(|| -> Result<Vec<f32>, DynError> {
                    in_chunk[..in_bytes]
                        .par_chunks(frame_bytes)
                        .try_fold(
                            || FftWorker::new(forward.clone(), half_bins),
                            |mut worker, raw_frame| {
                                decode_block_into_with_plan(
                                    raw_frame,
                                    &levels,
                                    cli.fft,
                                    &decode_plan,
                                    &mut worker.time,
                                    input_lsb,
                                )?;
                                worker.forward.process(&mut worker.time, &mut worker.spec)?;
                                for (dst, src) in worker.accum.iter_mut().zip(worker.spec.iter()) {
                                    *dst += src.norm_sqr();
                                }
                                Ok(worker)
                            },
                        )
                        .map(|res| res.map(|worker| worker.accum))
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
                .map_err(|e| IoError::other(e.to_string()))?;

            for (dst, src) in integrated.iter_mut().zip(chunk_accum.iter()) {
                *dst += *src;
            }

            done_frames += this_chunk as u64;
            if done_frames % cli.progress_every == 0 || done_frames == segment_frames {
                let pct = (done_frames as f64 / segment_frames as f64) * 100.0;
                print!(
                    "\r  progress [{}/{}]  : {done_frames}/{segment_frames} frames ({pct:.2}%)",
                    seg_idx + 1,
                    segment_total
                );
                std::io::stdout().flush()?;
                if done_frames == segment_frames {
                    println!();
                }
            }
        }

        let norm = integrated.iter().copied().sum::<f32>() * segment_frames as f32;
        let inv = if norm > 0.0 { 1.0f32 / norm } else { 1.0f32 };
        for v in &mut integrated {
            *v *= inv;
        }

        let raw_spectrum = Spectrum {
            freqs_mhz: freqs_mhz.clone(),
            values: integrated[start_bin..end_bin].to_vec(),
        };
        let (output_spectrum, calibration_rebin) = if let Some(table) = bandpass.as_ref() {
            let (calibrated, direction) = calibrate_bandpass(&raw_spectrum, table)
                .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
            println!("  bptable rebin    : {direction}");
            (calibrated, Some(direction))
        } else {
            (raw_spectrum, None)
        };
        let output_if_range = if calibration_rebin.is_some() {
            (
                output_spectrum.freqs_mhz[0],
                output_spectrum.freqs_mhz[output_spectrum.freqs_mhz.len() - 1],
            )
        } else {
            if_range_selected
        };

        let plot_path = resolve_plot_path(&cli.vdif, seg_idx, segment_total);
        plot_spectrum(
            &output_spectrum.freqs_mhz,
            &output_spectrum.values,
            &plot_path,
            output_if_range,
        )
        .map_err(|e| IoError::other(e.to_string()))?;
        println!("  plot             : {}", plot_path.display());

        if let Some(txt_path) = resolve_text_path(&cli.vdif, &cli.output, seg_idx, segment_total) {
            save_spectrum_text(
                &txt_path,
                &output_spectrum.freqs_mhz,
                &output_spectrum.values,
                &cli,
                output_if_range,
                segment_frames,
                calibration_rebin,
            )
            .map_err(|e| IoError::other(e.to_string()))?;
            println!("  output txt       : {}", txt_path.display());
        }
    }

    println!("  done             : {}", cli.vdif.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn spectrum(freqs_mhz: &[f32], values: &[f32]) -> Spectrum {
        Spectrum {
            freqs_mhz: freqs_mhz.to_vec(),
            values: values.to_vec(),
        }
    }

    fn assert_values_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn parses_output_table_with_comments() {
        let input = b"# spectrum\n1.0 2.0\n2.0 4.0 # inline\n";
        let parsed = parse_spectrum_table(Cursor::new(input), "test").unwrap();
        assert_values_close(&parsed.freqs_mhz, &[1.0, 2.0]);
        assert_values_close(&parsed.values, &[2.0, 4.0]);
    }

    #[test]
    fn divides_matching_channels_directly() {
        let data = spectrum(&[1.0, 2.0], &[10.0, 20.0]);
        let bandpass = spectrum(&[1.0, 2.0], &[2.0, 4.0]);
        let (calibrated, direction) = calibrate_bandpass(&data, &bandpass).unwrap();
        assert_eq!(direction, RebinDirection::None);
        assert_values_close(&calibrated.values, &[5.0, 5.0]);
    }

    #[test]
    fn rebins_fine_bandpass_to_coarse_data() {
        let data = spectrum(&[1.0, 3.0], &[10.0, 20.0]);
        let bandpass = spectrum(&[0.5, 1.5, 2.5, 3.5], &[2.0, 2.0, 4.0, 4.0]);
        let (calibrated, direction) = calibrate_bandpass(&data, &bandpass).unwrap();
        assert_eq!(direction, RebinDirection::BandpassToData);
        assert_values_close(&calibrated.freqs_mhz, &[1.0, 3.0]);
        assert_values_close(&calibrated.values, &[5.0, 5.0]);
    }

    #[test]
    fn rebins_fine_data_to_coarse_bandpass() {
        let data = spectrum(&[0.5, 1.5, 2.5, 3.5], &[10.0, 10.0, 20.0, 20.0]);
        let bandpass = spectrum(&[1.0, 3.0], &[2.0, 4.0]);
        let (calibrated, direction) = calibrate_bandpass(&data, &bandpass).unwrap();
        assert_eq!(direction, RebinDirection::DataToBandpass);
        assert_values_close(&calibrated.freqs_mhz, &[1.0, 3.0]);
        assert_values_close(&calibrated.values, &[5.0, 5.0]);
    }
}
