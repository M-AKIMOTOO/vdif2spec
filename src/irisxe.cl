__kernel void fft_init_bit_reverse(
    __global const float *input,
    __global float2 *output,
    __global const uint *bit_reverse,
    const uint fft_len)
{
    const size_t gid = get_global_id(0);
    const uint input_index = (uint)(gid % fft_len);
    const size_t frame_offset = gid - input_index;
    const uint reversed = bit_reverse[input_index];
    output[frame_offset + reversed] = (float2)(input[gid], 0.0f);
}

__kernel void fft_radix2_stage(
    __global const float2 *input,
    __global float2 *output,
    __global const float2 *twiddles,
    const uint fft_len,
    const uint stage_len)
{
    const size_t gid = get_global_id(0);
    const uint pairs_per_frame = fft_len >> 1;
    const uint pair = (uint)(gid % pairs_per_frame);
    // There are fft_len / 2 work items, but fft_len complex values, per frame.
    const size_t frame_offset = (gid - pair) * 2;
    const uint half_len = stage_len >> 1;
    const uint lane = pair % half_len;
    const uint group = pair / half_len;
    const size_t even_index = frame_offset + group * stage_len + lane;
    const size_t odd_index = even_index + half_len;
    const float2 twiddle = twiddles[(size_t)lane * (fft_len / stage_len)];
    const float2 even = input[even_index];
    const float2 odd = input[odd_index];
    const float2 twiddled = (float2)(
        twiddle.x * odd.x - twiddle.y * odd.y,
        twiddle.x * odd.y + twiddle.y * odd.x);
    output[even_index] = even + twiddled;
    output[odd_index] = even - twiddled;
}

__kernel void fft_integrate_power(
    __global const float2 *spectrum,
    __global float *integrated,
    const uint fft_len,
    const uint frame_count,
    const uint half_bins)
{
    const uint bin = (uint)get_global_id(0);
    if (bin >= half_bins) return;
    float total = 0.0f;
    for (uint frame = 0; frame < frame_count; ++frame) {
        const float2 value = spectrum[(size_t)frame * fft_len + bin];
        total += value.x * value.x + value.y * value.y;
    }
    integrated[bin] = total;
}
