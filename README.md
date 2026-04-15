# vdif2spec

VDIF/raw の周波数スペクトルを作成します。

- 復号は `phased_array` / `vdif_chN_spliter` と同系統（`--bit`, `--level`, `--shuffle`, `--sideband`）
- 周波数軸は `0 .. --bw/2 [MHz]`
- `--if LOW-HIGH` で IF 範囲を抽出
- `--cpu` で Rayon 並列実行

## Build

```bash
cargo build --release
```

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
- `--sideband <USB|LSB>`: 入力 sideband
- `--cpu <N>`: Rayon スレッド数
- `-o, --output [FILE]`: テキスト保存

## Output

PNG は常に出力されます:

- `<input_dir>/vdif2spec/<stem>_vdif2spec_spec.png`
- `--loop > 0` のときは `<stem>_vdif2spec_spec_loop0001.png` のように連番出力
- 生成後に `imagequant` で自動減色（indexed PNG 化）し、ファイル容量を最小化します

テキスト出力 (`frequency[MHz] power[a.u.]`) は `-o/--output` 指定時のみ:

- `-o`（引数なし）: `plot` と同じディレクトリ・同じ stem の `.txt`
- `-o <FILE>`: 指定ファイル名
- `--loop > 0` のときは `.txt` も `_loop0001` 連番になります

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
