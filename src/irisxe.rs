use std::ffi::{c_char, c_void, CString};
use std::mem::{size_of, transmute_copy};
use std::ptr;

type ClInt = i32;
type ClUInt = u32;
type ClULong = u64;
type ClPlatform = *mut c_void;
type ClDevice = *mut c_void;
type ClContext = *mut c_void;
type ClQueue = *mut c_void;
type ClMem = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_NOT_FOUND: ClInt = -1;
const CL_PLATFORM_NOT_FOUND_KHR: ClInt = -1001;
const CL_DEVICE_TYPE_GPU: ClULong = 1 << 2;
const CL_PLATFORM_NAME: ClUInt = 0x0902;
const CL_PLATFORM_VENDOR: ClUInt = 0x0903;
const CL_DEVICE_TYPE: ClUInt = 0x1000;
const CL_DEVICE_VENDOR_ID: ClUInt = 0x1001;
const CL_DEVICE_NAME: ClUInt = 0x102b;
const CL_DEVICE_HOST_UNIFIED_MEMORY: ClUInt = 0x1035;
const CL_CONTEXT_PLATFORM: isize = 0x1084;
const CL_MEM_READ_WRITE: ClULong = 1 << 0;
const CL_MEM_READ_ONLY: ClULong = 1 << 2;
const CL_PROGRAM_BUILD_LOG: ClUInt = 0x1183;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

type GetPlatformIds = unsafe extern "C" fn(ClUInt, *mut ClPlatform, *mut ClUInt) -> ClInt;
type GetPlatformInfo =
    unsafe extern "C" fn(ClPlatform, ClUInt, usize, *mut c_void, *mut usize) -> ClInt;
type GetDeviceIds =
    unsafe extern "C" fn(ClPlatform, ClULong, ClUInt, *mut ClDevice, *mut ClUInt) -> ClInt;
type GetDeviceInfo =
    unsafe extern "C" fn(ClDevice, ClUInt, usize, *mut c_void, *mut usize) -> ClInt;
type CreateContext = unsafe extern "C" fn(
    *const isize,
    ClUInt,
    *const ClDevice,
    Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>,
    *mut c_void,
    *mut ClInt,
) -> ClContext;
type CreateCommandQueue = unsafe extern "C" fn(ClContext, ClDevice, ClULong, *mut ClInt) -> ClQueue;
type CreateProgramWithSource = unsafe extern "C" fn(
    ClContext,
    ClUInt,
    *const *const c_char,
    *const usize,
    *mut ClInt,
) -> ClProgram;
type BuildProgram = unsafe extern "C" fn(
    ClProgram,
    ClUInt,
    *const ClDevice,
    *const c_char,
    Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
    *mut c_void,
) -> ClInt;
type GetProgramBuildInfo =
    unsafe extern "C" fn(ClProgram, ClDevice, ClUInt, usize, *mut c_void, *mut usize) -> ClInt;
type CreateKernel = unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel;
type CreateBuffer =
    unsafe extern "C" fn(ClContext, ClULong, usize, *mut c_void, *mut ClInt) -> ClMem;
type SetKernelArg = unsafe extern "C" fn(ClKernel, ClUInt, usize, *const c_void) -> ClInt;
type EnqueueWriteBuffer = unsafe extern "C" fn(
    ClQueue,
    ClMem,
    ClUInt,
    usize,
    usize,
    *const c_void,
    ClUInt,
    *const *mut c_void,
    *mut *mut c_void,
) -> ClInt;
type EnqueueReadBuffer = unsafe extern "C" fn(
    ClQueue,
    ClMem,
    ClUInt,
    usize,
    usize,
    *mut c_void,
    ClUInt,
    *const *mut c_void,
    *mut *mut c_void,
) -> ClInt;
type EnqueueNdRangeKernel = unsafe extern "C" fn(
    ClQueue,
    ClKernel,
    ClUInt,
    *const usize,
    *const usize,
    *const usize,
    ClUInt,
    *const *mut c_void,
    *mut *mut c_void,
) -> ClInt;
type Finish = unsafe extern "C" fn(ClQueue) -> ClInt;
type ReleaseMemObject = unsafe extern "C" fn(ClMem) -> ClInt;
type ReleaseKernel = unsafe extern "C" fn(ClKernel) -> ClInt;
type ReleaseProgram = unsafe extern "C" fn(ClProgram) -> ClInt;
type ReleaseCommandQueue = unsafe extern "C" fn(ClQueue) -> ClInt;
type ReleaseContext = unsafe extern "C" fn(ClContext) -> ClInt;

struct ClApi {
    handle: *mut c_void,
    get_platform_ids: GetPlatformIds,
    get_platform_info: GetPlatformInfo,
    get_device_ids: GetDeviceIds,
    get_device_info: GetDeviceInfo,
    create_context: CreateContext,
    create_command_queue: CreateCommandQueue,
    create_program_with_source: CreateProgramWithSource,
    build_program: BuildProgram,
    get_program_build_info: GetProgramBuildInfo,
    create_kernel: CreateKernel,
    create_buffer: CreateBuffer,
    set_kernel_arg: SetKernelArg,
    enqueue_write_buffer: EnqueueWriteBuffer,
    enqueue_read_buffer: EnqueueReadBuffer,
    enqueue_nd_range_kernel: EnqueueNdRangeKernel,
    finish: Finish,
    release_mem_object: ReleaseMemObject,
    release_kernel: ReleaseKernel,
    release_program: ReleaseProgram,
    release_command_queue: ReleaseCommandQueue,
    release_context: ReleaseContext,
}

impl ClApi {
    fn load() -> Result<Self, String> {
        let name = b"libOpenCL.so.1\0";
        let handle = unsafe { dlopen(name.as_ptr().cast(), 2) };
        if handle.is_null() {
            return Err("OpenCL loader libOpenCL.so.1 is not installed".into());
        }

        macro_rules! symbol {
            ($name:literal, $kind:ty) => {{
                let name = concat!($name, "\0");
                let value = unsafe { dlsym(handle, name.as_ptr().cast()) };
                if value.is_null() {
                    unsafe { dlclose(handle) };
                    return Err(format!("OpenCL loader is missing symbol {}", $name));
                }
                unsafe { transmute_copy::<*mut c_void, $kind>(&value) }
            }};
        }

        Ok(Self {
            handle,
            get_platform_ids: symbol!("clGetPlatformIDs", GetPlatformIds),
            get_platform_info: symbol!("clGetPlatformInfo", GetPlatformInfo),
            get_device_ids: symbol!("clGetDeviceIDs", GetDeviceIds),
            get_device_info: symbol!("clGetDeviceInfo", GetDeviceInfo),
            create_context: symbol!("clCreateContext", CreateContext),
            create_command_queue: symbol!("clCreateCommandQueue", CreateCommandQueue),
            create_program_with_source: symbol!(
                "clCreateProgramWithSource",
                CreateProgramWithSource
            ),
            build_program: symbol!("clBuildProgram", BuildProgram),
            get_program_build_info: symbol!("clGetProgramBuildInfo", GetProgramBuildInfo),
            create_kernel: symbol!("clCreateKernel", CreateKernel),
            create_buffer: symbol!("clCreateBuffer", CreateBuffer),
            set_kernel_arg: symbol!("clSetKernelArg", SetKernelArg),
            enqueue_write_buffer: symbol!("clEnqueueWriteBuffer", EnqueueWriteBuffer),
            enqueue_read_buffer: symbol!("clEnqueueReadBuffer", EnqueueReadBuffer),
            enqueue_nd_range_kernel: symbol!("clEnqueueNDRangeKernel", EnqueueNdRangeKernel),
            finish: symbol!("clFinish", Finish),
            release_mem_object: symbol!("clReleaseMemObject", ReleaseMemObject),
            release_kernel: symbol!("clReleaseKernel", ReleaseKernel),
            release_program: symbol!("clReleaseProgram", ReleaseProgram),
            release_command_queue: symbol!("clReleaseCommandQueue", ReleaseCommandQueue),
            release_context: symbol!("clReleaseContext", ReleaseContext),
        })
    }

    fn platform_string(&self, platform: ClPlatform, key: ClUInt) -> Result<String, String> {
        let mut size = 0usize;
        check(
            unsafe { (self.get_platform_info)(platform, key, 0, ptr::null_mut(), &mut size) },
            "query OpenCL platform string size",
        )?;
        let mut bytes = vec![0u8; size];
        check(
            unsafe {
                (self.get_platform_info)(
                    platform,
                    key,
                    bytes.len(),
                    bytes.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            },
            "query OpenCL platform string",
        )?;
        Ok(trim_opencl_string(bytes))
    }

    fn device_string(&self, device: ClDevice, key: ClUInt) -> Result<String, String> {
        let mut size = 0usize;
        check(
            unsafe { (self.get_device_info)(device, key, 0, ptr::null_mut(), &mut size) },
            "query OpenCL device string size",
        )?;
        let mut bytes = vec![0u8; size];
        check(
            unsafe {
                (self.get_device_info)(
                    device,
                    key,
                    bytes.len(),
                    bytes.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            },
            "query OpenCL device string",
        )?;
        Ok(trim_opencl_string(bytes))
    }
}

impl Drop for ClApi {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { dlclose(self.handle) };
        }
    }
}

pub struct IrisXeFft {
    api: ClApi,
    context: ClContext,
    queue: ClQueue,
    program: ClProgram,
    init_kernel: ClKernel,
    stage_kernel: ClKernel,
    power_kernel: ClKernel,
    input: ClMem,
    work_a: ClMem,
    work_b: ClMem,
    bit_reverse: ClMem,
    twiddles: ClMem,
    integrated: ClMem,
    fft_len: u32,
    log2_len: u32,
    half_bins: usize,
    batch_frames: usize,
    power_output: Vec<f32>,
}

impl IrisXeFft {
    pub fn new(fft_len: usize) -> Result<Self, String> {
        if fft_len < 2 || fft_len > u32::MAX as usize {
            return Err("--irisxe requires --fft between 2 and 2^32-1".into());
        }
        let log2_len = fft_len.trailing_zeros();
        let budget = 64usize * 1024 * 1024;
        let bytes_per_frame = fft_len
            .checked_mul(size_of::<f32>() * 5)
            .ok_or_else(|| "FFT buffer size overflow".to_string())?;
        if bytes_per_frame > budget {
            return Err(format!(
                "--fft {fft_len} needs more than the 64 MiB Iris Xe batch-buffer budget per frame"
            ));
        }
        let batch_frames = (budget / bytes_per_frame).clamp(1, 128);
        let half_bins = fft_len / 2;

        let api = ClApi::load()?;
        let (platform, device, platform_name, device_name) = select_iris_xe(&api)?;
        let context_properties = [CL_CONTEXT_PLATFORM, platform as isize, 0];
        let mut error = CL_SUCCESS;
        let context = unsafe {
            (api.create_context)(
                context_properties.as_ptr(),
                1,
                &device,
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        if context.is_null() || error != CL_SUCCESS {
            return Err(format!("create Intel GPU context failed ({error})"));
        }
        let queue = unsafe { (api.create_command_queue)(context, device, 0, &mut error) };
        if queue.is_null() || error != CL_SUCCESS {
            unsafe { (api.release_context)(context) };
            return Err(format!("create Intel GPU command queue failed ({error})"));
        }

        let source = include_bytes!("irisxe.cl");
        let source_ptr = source.as_ptr().cast::<c_char>();
        let source_len = source.len();
        let program = unsafe {
            (api.create_program_with_source)(context, 1, &source_ptr, &source_len, &mut error)
        };
        if program.is_null() || error != CL_SUCCESS {
            unsafe {
                (api.release_command_queue)(queue);
                (api.release_context)(context);
            }
            return Err(format!("create OpenCL FFT program failed ({error})"));
        }
        let options = CString::new("-cl-std=CL1.2").expect("static OpenCL option");
        error = unsafe {
            (api.build_program)(program, 1, &device, options.as_ptr(), None, ptr::null_mut())
        };
        if error != CL_SUCCESS {
            let log = program_build_log(&api, program, device).unwrap_or_default();
            unsafe {
                (api.release_program)(program);
                (api.release_command_queue)(queue);
                (api.release_context)(context);
            }
            return Err(format!(
                "compile Iris Xe FFT kernels failed ({error}):\n{log}"
            ));
        }

        let init_kernel = create_kernel(&api, program, "fft_init_bit_reverse")?;
        let stage_kernel = match create_kernel(&api, program, "fft_radix2_stage") {
            Ok(kernel) => kernel,
            Err(error) => {
                unsafe { (api.release_kernel)(init_kernel) };
                release_program_queue_context(&api, program, queue, context);
                return Err(error);
            }
        };
        let power_kernel = match create_kernel(&api, program, "fft_integrate_power") {
            Ok(kernel) => kernel,
            Err(error) => {
                unsafe {
                    (api.release_kernel)(stage_kernel);
                    (api.release_kernel)(init_kernel);
                }
                release_program_queue_context(&api, program, queue, context);
                return Err(error);
            }
        };

        let input_bytes = batch_frames
            .checked_mul(fft_len)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| "Intel GPU input buffer size overflow".to_string())?;
        let work_bytes = batch_frames
            .checked_mul(fft_len)
            .and_then(|n| n.checked_mul(size_of::<[f32; 2]>()))
            .ok_or_else(|| "Intel GPU FFT buffer size overflow".to_string())?;
        let power_bytes = half_bins
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| "Intel GPU output buffer size overflow".to_string())?;
        let reverse_bytes = fft_len
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| "Intel GPU bit-reversal LUT size overflow".to_string())?;
        let twiddle_bytes = half_bins
            .checked_mul(size_of::<[f32; 2]>())
            .ok_or_else(|| "Intel GPU twiddle LUT size overflow".to_string())?;
        let input = create_buffer(&api, context, input_bytes)?;
        let work_a = create_buffer(&api, context, work_bytes)?;
        let work_b = create_buffer(&api, context, work_bytes)?;
        let bit_reverse = create_buffer_with_flags(&api, context, reverse_bytes, CL_MEM_READ_ONLY)?;
        let twiddles = create_buffer_with_flags(&api, context, twiddle_bytes, CL_MEM_READ_ONLY)?;
        let integrated = create_buffer(&api, context, power_bytes)?;
        let reverse_lut: Vec<u32> = (0..fft_len)
            .map(|index| (index as u32).reverse_bits() >> (u32::BITS - log2_len))
            .collect();
        let twiddle_lut: Vec<[f32; 2]> = (0..half_bins)
            .map(|index| {
                let angle = -std::f32::consts::TAU * index as f32 / fft_len as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        write_buffer(
            &api,
            queue,
            bit_reverse,
            &reverse_lut,
            "upload bit-reversal LUT",
        )?;
        write_buffer(
            &api,
            queue,
            twiddles,
            &twiddle_lut,
            "upload FFT twiddle LUT",
        )?;

        println!("  GPU              : {device_name} ({platform_name})");
        println!("  GPU batch frames : {batch_frames}");
        Ok(Self {
            api,
            context,
            queue,
            program,
            init_kernel,
            stage_kernel,
            power_kernel,
            input,
            work_a,
            work_b,
            bit_reverse,
            twiddles,
            integrated,
            fft_len: fft_len as u32,
            log2_len,
            half_bins,
            batch_frames,
            power_output: vec![0.0; half_bins],
        })
    }

    pub fn batch_frames(&self) -> usize {
        self.batch_frames
    }

    pub fn process(&mut self, samples: &[f32], frames: usize) -> Result<&[f32], String> {
        if frames == 0 || frames > self.batch_frames {
            return Err(format!(
                "invalid Iris Xe batch size {frames} (maximum {})",
                self.batch_frames
            ));
        }
        let expected = frames
            .checked_mul(self.fft_len as usize)
            .ok_or_else(|| "Iris Xe input size overflow".to_string())?;
        if samples.len() != expected {
            return Err(format!(
                "Iris Xe input has {} samples; expected {expected}",
                samples.len()
            ));
        }
        let bytes = std::mem::size_of_val(samples);
        check(
            unsafe {
                (self.api.enqueue_write_buffer)(
                    self.queue,
                    self.input,
                    1,
                    0,
                    bytes,
                    samples.as_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "upload decoded samples to Iris Xe",
        )?;

        set_arg(&self.api, self.init_kernel, 0, &self.input)?;
        set_arg(&self.api, self.init_kernel, 1, &self.work_a)?;
        set_arg(&self.api, self.init_kernel, 2, &self.bit_reverse)?;
        set_arg(&self.api, self.init_kernel, 3, &self.fft_len)?;
        enqueue_kernel(
            &self.api,
            self.queue,
            self.init_kernel,
            frames * self.fft_len as usize,
        )?;

        let mut source = self.work_a;
        let mut destination = self.work_b;
        for stage in 1..=self.log2_len {
            let stage_len = 1u32 << stage;
            set_arg(&self.api, self.stage_kernel, 0, &source)?;
            set_arg(&self.api, self.stage_kernel, 1, &destination)?;
            set_arg(&self.api, self.stage_kernel, 2, &self.twiddles)?;
            set_arg(&self.api, self.stage_kernel, 3, &self.fft_len)?;
            set_arg(&self.api, self.stage_kernel, 4, &stage_len)?;
            enqueue_kernel(
                &self.api,
                self.queue,
                self.stage_kernel,
                frames * (self.fft_len as usize / 2),
            )?;
            std::mem::swap(&mut source, &mut destination);
        }

        let frame_count = frames as u32;
        let half_bins = self.half_bins as u32;
        set_arg(&self.api, self.power_kernel, 0, &source)?;
        set_arg(&self.api, self.power_kernel, 1, &self.integrated)?;
        set_arg(&self.api, self.power_kernel, 2, &self.fft_len)?;
        set_arg(&self.api, self.power_kernel, 3, &frame_count)?;
        set_arg(&self.api, self.power_kernel, 4, &half_bins)?;
        enqueue_kernel(&self.api, self.queue, self.power_kernel, self.half_bins)?;

        check(
            unsafe {
                (self.api.enqueue_read_buffer)(
                    self.queue,
                    self.integrated,
                    1,
                    0,
                    self.power_output.len() * size_of::<f32>(),
                    self.power_output.as_mut_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "read integrated spectrum from Iris Xe",
        )?;
        Ok(&self.power_output)
    }
}

impl Drop for IrisXeFft {
    fn drop(&mut self) {
        unsafe {
            (self.api.finish)(self.queue);
            (self.api.release_mem_object)(self.integrated);
            (self.api.release_mem_object)(self.twiddles);
            (self.api.release_mem_object)(self.bit_reverse);
            (self.api.release_mem_object)(self.work_b);
            (self.api.release_mem_object)(self.work_a);
            (self.api.release_mem_object)(self.input);
            (self.api.release_kernel)(self.power_kernel);
            (self.api.release_kernel)(self.stage_kernel);
            (self.api.release_kernel)(self.init_kernel);
            (self.api.release_program)(self.program);
            (self.api.release_command_queue)(self.queue);
            (self.api.release_context)(self.context);
        }
    }
}

fn select_iris_xe(api: &ClApi) -> Result<(ClPlatform, ClDevice, String, String), String> {
    let mut count = 0;
    let status = unsafe { (api.get_platform_ids)(0, ptr::null_mut(), &mut count) };
    if status == CL_PLATFORM_NOT_FOUND_KHR || count == 0 {
        return Err("no OpenCL platforms found; Intel GPU compute runtime is unavailable".into());
    }
    check(status, "enumerate OpenCL platforms")?;
    let mut platforms = vec![ptr::null_mut(); count as usize];
    check(
        unsafe { (api.get_platform_ids)(count, platforms.as_mut_ptr(), ptr::null_mut()) },
        "enumerate OpenCL platforms",
    )?;

    let mut candidates = Vec::new();
    for platform in platforms {
        let vendor = api.platform_string(platform, CL_PLATFORM_VENDOR)?;
        let platform_name = api.platform_string(platform, CL_PLATFORM_NAME)?;
        if !vendor.to_ascii_lowercase().contains("intel")
            && !platform_name.to_ascii_lowercase().contains("intel")
        {
            continue;
        }
        let mut device_count = 0;
        let status = unsafe {
            (api.get_device_ids)(
                platform,
                CL_DEVICE_TYPE_GPU,
                0,
                ptr::null_mut(),
                &mut device_count,
            )
        };
        if status == CL_DEVICE_NOT_FOUND || device_count == 0 {
            continue;
        }
        check(status, "enumerate Intel GPU devices")?;
        let mut devices = vec![ptr::null_mut(); device_count as usize];
        check(
            unsafe {
                (api.get_device_ids)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    device_count,
                    devices.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "enumerate Intel GPU devices",
        )?;

        for device in devices {
            let mut vendor_id = 0u32;
            let mut device_type = 0u64;
            let mut unified_memory = 0u32;
            check(
                unsafe {
                    (api.get_device_info)(
                        device,
                        CL_DEVICE_VENDOR_ID,
                        size_of::<u32>(),
                        (&mut vendor_id as *mut u32).cast(),
                        ptr::null_mut(),
                    )
                },
                "query Intel GPU vendor ID",
            )?;
            check(
                unsafe {
                    (api.get_device_info)(
                        device,
                        CL_DEVICE_TYPE,
                        size_of::<u64>(),
                        (&mut device_type as *mut u64).cast(),
                        ptr::null_mut(),
                    )
                },
                "query Intel GPU device type",
            )?;
            if vendor_id != 0x8086 || device_type & CL_DEVICE_TYPE_GPU == 0 {
                continue;
            }
            let unified_status = unsafe {
                (api.get_device_info)(
                    device,
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    size_of::<u32>(),
                    (&mut unified_memory as *mut u32).cast(),
                    ptr::null_mut(),
                )
            };
            let device_name = api.device_string(device, CL_DEVICE_NAME)?;
            let name_lower = device_name.to_ascii_lowercase();
            let looks_integrated = unified_status == CL_SUCCESS && unified_memory != 0;
            if !looks_integrated && !name_lower.contains("iris") {
                continue;
            }
            let score = (name_lower.contains("iris") as i32) * 100
                + (name_lower.contains("xe") as i32) * 10
                + looks_integrated as i32;
            candidates.push((score, platform, device, platform_name.clone(), device_name));
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    candidates
        .into_iter()
        .next()
        .map(|(_, platform, device, platform_name, device_name)| {
            (platform, device, platform_name, device_name)
        })
        .ok_or_else(|| {
            "no Intel integrated GPU OpenCL device found; install the Intel Graphics Compute Runtime (OpenCL ICD)".into()
        })
}

fn create_kernel(api: &ClApi, program: ClProgram, name: &str) -> Result<ClKernel, String> {
    let name = CString::new(name).map_err(|e| e.to_string())?;
    let mut error = CL_SUCCESS;
    let kernel = unsafe { (api.create_kernel)(program, name.as_ptr(), &mut error) };
    if kernel.is_null() || error != CL_SUCCESS {
        return Err(format!("create OpenCL kernel failed ({error})"));
    }
    Ok(kernel)
}

fn create_buffer(api: &ClApi, context: ClContext, size: usize) -> Result<ClMem, String> {
    create_buffer_with_flags(api, context, size, CL_MEM_READ_WRITE)
}

fn create_buffer_with_flags(
    api: &ClApi,
    context: ClContext,
    size: usize,
    flags: ClULong,
) -> Result<ClMem, String> {
    let mut error = CL_SUCCESS;
    let buffer = unsafe { (api.create_buffer)(context, flags, size, ptr::null_mut(), &mut error) };
    if buffer.is_null() || error != CL_SUCCESS {
        return Err(format!(
            "allocate Intel GPU buffer ({size} bytes) failed ({error})"
        ));
    }
    Ok(buffer)
}

fn write_buffer<T>(
    api: &ClApi,
    queue: ClQueue,
    buffer: ClMem,
    values: &[T],
    operation: &str,
) -> Result<(), String> {
    check(
        unsafe {
            (api.enqueue_write_buffer)(
                queue,
                buffer,
                1,
                0,
                std::mem::size_of_val(values),
                values.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        },
        operation,
    )
}

fn program_build_log(api: &ClApi, program: ClProgram, device: ClDevice) -> Result<String, String> {
    let mut size = 0usize;
    check(
        unsafe {
            (api.get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null_mut(),
                &mut size,
            )
        },
        "query OpenCL compiler log size",
    )?;
    let mut bytes = vec![0u8; size];
    check(
        unsafe {
            (api.get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                bytes.len(),
                bytes.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
        },
        "query OpenCL compiler log",
    )?;
    Ok(trim_opencl_string(bytes))
}

fn release_program_queue_context(
    api: &ClApi,
    program: ClProgram,
    queue: ClQueue,
    context: ClContext,
) {
    unsafe {
        (api.release_program)(program);
        (api.release_command_queue)(queue);
        (api.release_context)(context);
    }
}

fn set_arg<T>(api: &ClApi, kernel: ClKernel, index: ClUInt, value: &T) -> Result<(), String> {
    check(
        unsafe { (api.set_kernel_arg)(kernel, index, size_of::<T>(), (value as *const T).cast()) },
        "set OpenCL kernel argument",
    )
}

fn enqueue_kernel(
    api: &ClApi,
    queue: ClQueue,
    kernel: ClKernel,
    global_size: usize,
) -> Result<(), String> {
    check(
        unsafe {
            (api.enqueue_nd_range_kernel)(
                queue,
                kernel,
                1,
                ptr::null(),
                &global_size,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        },
        "enqueue Iris Xe FFT kernel",
    )
}

fn check(status: ClInt, operation: &str) -> Result<(), String> {
    if status == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed with OpenCL error {status}"))
    }
}

fn trim_opencl_string(mut bytes: Vec<u8>) -> String {
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use realfft::RealFftPlanner;

    #[test]
    #[ignore = "requires an Intel integrated GPU with an OpenCL runtime"]
    fn batched_gpu_power_matches_cpu_fft() {
        for fft_len in [16, 256, 16384] {
            let mut gpu = IrisXeFft::new(fft_len).unwrap();
            let mut planner = RealFftPlanner::<f32>::new();
            let forward = planner.plan_fft_forward(fft_len);
            let mut spectrum = forward.make_output_vec();
            let mut scratch = forward.make_scratch_vec();
            for frames in [gpu.batch_frames(), 3, 1] {
                let frames = frames.min(gpu.batch_frames());
                let samples: Vec<f32> = (0..fft_len * frames)
                    .map(|i| ((i * 37 + i / fft_len * 13) % 101) as f32 / 31.0 - 1.5)
                    .collect();
                let mut expected = vec![0.0f32; fft_len / 2];
                for frame in samples.chunks_exact(fft_len) {
                    let mut time = frame.to_vec();
                    forward
                        .process_with_scratch(&mut time, &mut spectrum, &mut scratch)
                        .unwrap();
                    for (power, bin) in expected.iter_mut().zip(&spectrum) {
                        *power += bin.norm_sqr();
                    }
                }
                let actual = gpu.process(&samples, frames).unwrap();
                for (bin, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() < 0.05 + expected.abs() * 1.0e-4,
                        "FFT={fft_len}, frames={frames}, bin={bin}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
