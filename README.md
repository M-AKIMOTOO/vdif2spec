# vdif2spec

VDIF/raw の周波数スペクトルを作成します。

- 復号は `phased_array` / `vdif_chN_spliter` と同系統（`--bit`, `--level`, `--shuffle`, `--sideband`）
- 周波数軸は `0 .. --bw/2 [MHz]`
- `--if LOW-HIGH` で IF 範囲を抽出
- `--bptable FILE` でチャンネルごとのバンドパス較正
- `--cpu` で Rayon 並列実行（既定値は OS が報告する利用可能スレッド数）

## Build

```bash
cargo build --release
```

`.cargo/config.toml` の `target-cpu=native` により、ビルドした PC の CPU を自動検出して最適化します。CPU 名や命令セットを固定していないため、各 PC 上で同じコマンドを実行できます。生成したバイナリはその CPU 向けなので、別の PC ではその PC 上で再ビルドしてください。

## Usage

```bash
vdif2spec --vdif <FILE> [OPTIONS]
```

主要オプション:

- `--vdif <FILE>`: 入力 VDIF/raw
- `--fft <N>`: FFT 点数（2 のべき乗）
- `--skip <SEC>`: 先頭からスキップ秒
- `--length <SEC>`: 処理時間
- `--loop <N>`: `--length` 秒ごとの出力セグメント数（`0` で単一出力）
- `--bit <BITS>`: 量子化 bit 数
- `--level <L0> <L1> ...`: 量子化レベル（code order、個数は `2^bit`）
- `--shuffle <B31> ... <B0>`: ビットシャッフル（32 個、`0..31` の置換）
- `--vsrec`: `--level`/`--shuffle` を VSREC 固定値へ強制上書き
- `--bw <MHz>`: サンプリング帯域（スペクトル表示は `0..bw/2`）
- `--if <LOW> <HIGH>`: IF 抽出範囲
- `--bptable <FILE>`: 2列テキストのバンドパステーブルで `power / bandpass` 較正
- `--sideband <USB|LSB>`: 入力 sideband
- `--cpu <N>`: Rayon スレッド数（既定値は利用可能な並列数）
- `--irisxe`: Intel 統合 GPU を OpenCL で選択し、バッチ FFT を実行（Intel OpenCL ICD が必要）
- `--no-plot`: PNG描画を省略し、`--output` のテキストだけを保存
- `-o, --output [FILE]`: テキスト保存

## Output

通常は PNG を出力します。`--no-plot` を指定すると PNG 描画と減色処理を省略できます（この場合 `--output` が必要です）:

- `<input_dir>/vdif2spec/<stem>_vdif2spec_spec.png`
- `--loop > 0` のときは `<stem>_vdif2spec_spec_loop0001.png` のように連番出力
- 生成後に `imagequant` で減色（indexed PNG 化）します。処理速度を優先する設定のため、圧縮後のファイルは大きくなる場合があります

テキスト出力 (`frequency[MHz] power[a.u.]`) は `-o/--output` 指定時のみ:

- `-o`（引数なし）: `plot` と同じディレクトリ・同じ stem の `.txt`
- `-o <FILE>`: 指定ファイル名
- `--loop > 0` のときは `.txt` も `_loop0001` 連番になります

## Bandpass calibration

`--bptable` は `--output` で生成したものと同じ、空白区切りの2列テキストを読み込みます。`#` から始まる行と行末コメントは無視されます。

```text
# Frequency(MHz) Power(a.u.)
60.000000000 1.234000000e-5
60.250000000 1.210000000e-5
```

較正値は各チャンネルの `power / bandpass` です。データとテーブルのチャンネル幅が異なる場合は、細かい側を周波数ビンの重なり幅で加重平均し、粗い側のグリッドへ rebin します。出力は両者が完全に重なる周波数チャンネルに限定されます。

## Examples

全帯域（`0..bw/2`）:

```bash
./target/release/vdif2spec \
  --vdif ./test/HITACH32_2024058103800.vdif2spec \
  --fft 4096 --bw 1024 --bit 2 --sideband LSB --cpu 4
```

IF 抽出 + 自動テキスト名（plot と同じ場所）:

```bash
./target/release/vdif2spec \
  --vdif ./test/HITACH32_2024058103800.vdif2spec \
  --fft 4096 --bw 1024 --if 60 100 --length 1 --loop 5 --cpu 4 -o
```

IF 抽出 + 任意テキスト名:

```bash
./target/release/vdif2spec \
  --vdif ./test/HITACH32_2024058103800.vdif2spec \
  --fft 4096 --bw 1024 --if 60 100 --output ./spec.txt
```

バンドパス較正 + テキスト出力:

```bash
./target/release/vdif2spec \
  --vdif ./test/HITACH32_2024058103800.vdif2spec \
  --fft 4096 --bw 1024 --if 60 100 \
  --bptable ./bandpass.txt --output ./calibrated.txt
```

テキストのみ高速出力:

```bash
./target/release/vdif2spec \
  --vdif ./test/HITACH32_2024058103800.vdif2spec \
  --fft 4096 --bw 1024 --if 60 100 \
  --bptable ./bandpass.txt --output ./calibrated.txt --no-plot
```

Iris Xe での計算:

```bash
./target/release/vdif2spec \
  --vdif ./input.vdif --fft 16384 --irisxe --no-plot --output ./spectrum.txt
```

`--irisxe` は Intel 製の統合 GPU を OpenCL デバイスから選択し、CPU で復号した f32 サンプルを小分けで転送して FFT と電力積算を実行します。OpenCL の Intel Graphics Compute Runtime が必要です。通常の CPU 実行には追加ランタイムは不要です。`switcherooctl --gpu=0` は OpenGL 描画用の GPU 選択であり、この計算バックエンドは Intel OpenCL デバイスを直接選びます。

## CPU performance

CPU 実行では復号・FFT・積算のバッファをワーカーごとに再利用し、FFT ごとの作業メモリ確保を省いています。1/2/4/8 bit の復号はバイトごとの固定長コピーを使い、VSREC のビット反転も復号テーブルに組み込みます。16 フレーム単位で仕事を分配してコア間の負荷を調整します。入力バッファは処理対象のフレーム数と 64 MiB を上限とします（1 FFT フレームが 64 MiB を超える場合は 1 フレーム分）。

2026-11-10、Core i7-12700H、`NOBEYA45_2024131043000.raw` 全体（1.28 GB、5 秒分）、`--fft 16384 --bit 2 --no-plot` での計測結果です。変更後は `target-cpu=native` を有効にしています。ファイルキャッシュを温め、変更前後を交互に 5 回実行した中央値を示します。PNG 描画・減色の時間は含みません。

| CPU threads | Before | After | Speedup |
| --- | --- | --- | --- |
| 4 | 3.549 s | 2.098 s | 1.69x |
| 20 | 2.974 s | 1.396 s | 2.13x |

この比較での出力電力の最大相対差は `5.9e-7` 未満でした。並列積算の加算順序が変わるため、浮動小数点の丸めによる差が生じます。

比較は Python 標準ライブラリだけで再実行できます。`--before` には変更前に保存した release バイナリを指定します。

```bash
python3 scripts/bench_cpu.py \
  --before /tmp/vdif2spec-before --after ./target/release/vdif2spec \
  --vdif /path/to/input.raw --fft 16384 --cpu 4 20 \
  --length 5 --repeat 5
```

計測時間とスペクトルを一時ディレクトリへ保存し、周波数の一致、電力の相対誤差 `1e-5`（絶対誤差 `1e-20`）以内を確認します。`--directory` で保存先、`--vsrec` で VSREC 復号を指定できます。

Intel GPU のバッチ FFT と CPU FFT の比較テストは、Intel OpenCL ランタイムを利用できる環境で実行してください。

```bash
cargo test --release batched_gpu_power_matches_cpu_fft -- --ignored --nocapture
```
