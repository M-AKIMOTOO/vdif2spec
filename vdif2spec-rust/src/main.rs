use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use chrono::{NaiveDateTime, TimeZone, Local, Duration};
use plotters::prelude::*;
use clap::Parser;

/// コマンドライン引数をパースするための構造体
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// プロットするファイルの名前
    #[arg(short, long)]
    tsys: String,

    #[arg(short, long)]
    skd: String,
}

fn read_sked_data(file_path: &str) -> Result<Vec<(String, NaiveDateTime, f64)>, Box<dyn Error>> {
    let path = Path::new(file_path);
    if !path.exists(){
        println!("{} が存在しません。", file_path);
        return Ok(Vec::new());
    }
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    let mut in_sked_section = false;
    for line in reader.lines() {
        let line = line?;
        if line.starts_with("$SKED") {
            in_sked_section = true;
            continue;
        }
        if in_sked_section {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let source = parts[0].to_string();
            let time_str = parts[1].to_string();
            let elapsed_time = parts[2].parse::<f64>()?;
            let year = time_str[0..2].parse::<i32>()? + 2000; //西暦2000年以降を想定
            let day_of_year = time_str[2..5].parse::<u32>()?;
            let hour = time_str[5..7].parse::<u32>()?;
            let minute = time_str[7..9].parse::<u32>()?;
            let second = time_str[9..11].parse::<u32>()?;
            let date_time = Local.ymd(year,1,1).and_hms(0,0,0) + Duration::days((day_of_year - 1) as i64) + Duration::hours(hour as i64) + Duration::minutes(minute as i64) + Duration::seconds(second as i64);

            data.push((source, date_time.naive_local(), elapsed_time));
        }
    }

    Ok(data)
}

fn read_data_from_file(file_path: &str) -> Result<Vec<(NaiveDateTime, Vec<String>)>, Box<dyn Error>> {
    let path = Path::new(file_path);
    if !path.exists(){
        println!("{} が存在しません。", file_path);
        return Ok(Vec::new());
    }
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with("#") {
            continue; // ヘッダー行はスキップ
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue; // データが不足している行はスキップ
        }

        // 日付と時刻をパース
        let datetime_str = format!("{} {}", parts[0], parts[1]);
        let date_time = NaiveDateTime::parse_from_str(&datetime_str, "%d/%b/%Y %H:%M:%S")?;

        let mut row = Vec::new();
        // power 値をパース
        for part in &parts[2..] {
            row.push(part.to_string());
        }
        data.push((date_time,row));
    }

    Ok(data)
}

fn plot_power(data: &Vec<(f64, Vec<f64>)>, power_index: usize, start: NaiveDateTime, end: NaiveDateTime, output_dir: &Path,tsys_filename:&str,source:&str) -> Result<(), Box<dyn Error>> {
    // ファイル名を生成
    let filename = format!("{}_{}_{}_ch{}_tsys2plot.png",tsys_filename,source,start.format("%Y%m%d%H%M%S").to_string(),power_index + 1);
    let file_path = output_dir.join(filename); // 出力ディレクトリとファイル名を結合

    let root_area = BitMapBackend::new(&file_path, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let min_time = 0.0;
    let max_time = (end - start).num_seconds() as f64;

    let (min_power, max_power) = data
        .iter()
        .map(|(_,powers)| powers[power_index])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), p| {
            (min.min(p), max.max(p))
        });

    let mut chart = ChartBuilder::on(&root_area)
        .margin(5)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(min_time..max_time, min_power..max_power)?;

    chart.configure_mesh()
        .axis_style(&BLACK)
        .x_labels(10)
        .y_labels(10)
        .x_label_style(("sans-serif", 18).into_font())
        .y_label_style(("sans-serif", 18).into_font())
        .x_desc(&format!("Elapsed time since {} UT (sec)", start.format("%Y-%m-%d %H:%M:%S").to_string()))
        .y_desc(&format!("Ch {} (dBm)", power_index + 1))
        .draw()?;

    let series_data: Vec<(f64, f64)> = data
        .iter()
        .map(|(elapsed_time,powers)| (*elapsed_time, powers[power_index]))
        .collect();

    chart.draw_series(LineSeries::new(series_data, &RED).point_size(0))?;

    root_area.present()?;
    println!("   Generating >>> {}", file_path.to_str().unwrap()); // 生成されたファイルパスを表示
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // コマンドライン引数からファイルパスを取得
    let file_path = &args.tsys;
    let sked_file_path = &args.skd;
    
    // tsys のファイル名を格納
    let tsys_filename = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default");

    // 出力ディレクトリ名を作成
    let sked_filename = Path::new(sked_file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default");
    let output_dir_name = format!("{}_tsys2plot", sked_filename);
    let output_dir = Path::new(&output_dir_name);

    // 出力ディレクトリが存在しない場合は作成
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir)?;
    }

    // ファイルからデータを読み込む
    let tsys_data = read_data_from_file(file_path)?;
    let sked_data = read_sked_data(sked_file_path)?;

    //sked_dataをsource毎にわける
    let mut sked_data_by_source: std::collections::HashMap<String, Vec<(NaiveDateTime,f64)>> = std::collections::HashMap::new();
    for (source,start,elapsed) in sked_data{
        sked_data_by_source.entry(source).or_insert_with(Vec::new).push((start,elapsed));
    }

    // 天体ごとに処理
    for (source, sked_entries) in sked_data_by_source {
        println!("Processing source: {}", source);
        for (start, elapsed_time) in sked_entries {
            let end = start + Duration::minutes((elapsed_time / 60.0) as i64);

            // 該当範囲のデータを抽出
            let filtered_data: Vec<(f64,Vec<f64>)> = tsys_data
                .iter()
                .filter_map(|(row_datetime,powers)| {
                    if *row_datetime >= start && *row_datetime <= end {
                        let elapsed_time = (*row_datetime - start).num_seconds() as f64;
                        let powers_f64: Vec<f64> = powers.iter().filter_map(|p| p.parse::<f64>().ok()).collect();
                        if powers_f64.len() == powers.len(){
                           Some((elapsed_time,powers_f64))
                        }else{
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            // powerごとにプロット
            let num_powers = if filtered_data.len() > 0 && filtered_data[0].1.len() > 0 { filtered_data[0].1.len() } else { 0 };
             if num_powers > 0 {
                // 天体用のディレクトリを作成
                let source_dir = output_dir.join(&source);
                if !source_dir.exists(){
                    fs::create_dir_all(&source_dir)?;
                }
                for power_index in 0..num_powers {
                    plot_power(&filtered_data, power_index, start, end, &source_dir,tsys_filename,&source)?;
                }
            } else {
                println!("No power data for this range.");
            }
        }
    }

    Ok(())
}
