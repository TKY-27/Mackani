//! Facts about this machine, read at run time with read-only `sysctlbyname`
//! queries and one IORegistry property read (no setting is changed). A missing name is `None`, never an error:
//! callers choose a fallback or fail closed.

/// What this process can learn about the machine it runs on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Machine {
    /// `machdep.cpu.brand_string`, e.g. "Apple M5".
    pub chip: Option<String>,
    /// `kern.osproductversion`, e.g. "27.2.1".
    pub macos_version: Option<String>,
    /// Physical cores of each performance level, fastest first
    /// (`hw.perflevel{i}.physicalcpu` for i < `hw.nperflevels`); empty when unknown.
    pub core_levels: Vec<usize>,
    /// `hw.physicalcpu`.
    pub physical_cores: Option<usize>,
    /// CPUs this process may use (`std::thread::available_parallelism`).
    pub available_cpus: Option<usize>,
    /// `hw.memsize` in bytes.
    pub memory_bytes: Option<u64>,
    /// `hw.pagesize` in bytes.
    pub page_size: Option<u64>,
    /// GPU cores: `gpu-core-count` of the IORegistry's `AGXAccelerator`
    /// entry, read with IOKit (no setting is changed).
    pub gpu_cores: Option<usize>,
}

impl Machine {
    /// Read available hardware facts, leaving unavailable values unspecified.
    pub fn detect() -> Self {
        let levels = sysctl_u64("hw.nperflevels").unwrap_or(0);
        let core_levels: Option<Vec<usize>> = (0..levels)
            .map(|level| {
                sysctl_u64(&format!("hw.perflevel{level}.physicalcpu"))
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
            })
            .collect();
        Self {
            chip: sysctl_string("machdep.cpu.brand_string"),
            macos_version: sysctl_string("kern.osproductversion"),
            // One missing level makes the split unknown rather than wrong.
            core_levels: core_levels.unwrap_or_default(),
            physical_cores: sysctl_u64("hw.physicalcpu")
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0),
            available_cpus: std::thread::available_parallelism().ok().map(|n| n.get()),
            memory_bytes: sysctl_u64("hw.memsize").filter(|&n| n > 0),
            page_size: sysctl_u64("hw.pagesize").filter(|&n| n > 0),
            gpu_cores: gpu_core_count(),
        }
    }

    /// One CPU worker per physical core of every performance level (on the M5,
    /// 4 + 6 = 10, the count measured best), falling back to `hw.physicalcpu`,
    /// then to the CPUs available to the process, then to 1; never more than the
    /// process may use, or [`Self::MAX_UNBOUNDED_THREADS`] when that is unknown.
    /// A zero or overflowing count is ignored like a missing one.
    pub fn default_threads(&self) -> usize {
        let levels = self
            .core_levels
            .iter()
            .try_fold(0_usize, |sum, &cores| sum.checked_add(cores));
        let cores = [levels, self.physical_cores, self.available_cpus]
            .into_iter()
            .flatten()
            .find(|&cores| cores > 0)
            .unwrap_or(1);
        cores
            .min(self.available_cpus.unwrap_or(Self::MAX_UNBOUNDED_THREADS))
            .max(1)
    }

    /// The most default workers when the CPUs available cannot be read: a
    /// sane bound for an absurd sysctl value, not a measured optimum.
    pub const MAX_UNBOUNDED_THREADS: usize = 64;
}

/// An integer sysctl (4 or 8 bytes, little-endian), `None` when it does not exist.
pub fn sysctl_u64(name: &str) -> Option<u64> {
    let bytes = sysctl_bytes(name)?;
    match bytes.len() {
        4 => Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?))),
        8 => Some(u64::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

/// A string sysctl without its terminating NUL, `None` when it does not exist.
pub fn sysctl_string(name: &str) -> Option<String> {
    let bytes = sysctl_bytes(name)?;
    let text = bytes.split(|&b| b == 0).next()?;
    let text = String::from_utf8(text.to_vec()).ok()?;
    (!text.is_empty()).then_some(text)
}

#[cfg(target_os = "macos")]
fn sysctl_bytes(name: &str) -> Option<Vec<u8>> {
    use std::ffi::{CString, c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let name = CString::new(name).ok()?;
    let mut len = 0_usize;
    // SAFETY: `name` is NUL-terminated; a null `oldp` asks only for the size,
    // written to the live `len`. `newp` is null with length 0: nothing is set.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || len == 0 {
        return None;
    }
    let mut buffer = vec![0_u8; len];
    // SAFETY: `buffer` has `len` writable bytes and `len` says so; the kernel
    // writes at most that many and stores the actual length. Read-only as above.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return None;
    }
    buffer.truncate(len);
    Some(buffer)
}

#[cfg(not(target_os = "macos"))]
fn sysctl_bytes(_name: &str) -> Option<Vec<u8>> {
    None
}

/// `gpu-core-count` of the first IORegistry entry of class `AGXAccelerator`
/// (the Apple GPU), `None` when there is no such entry or property. Metal
/// and sysctl do not report the GPU core count.
#[cfg(target_os = "macos")]
fn gpu_core_count() -> Option<usize> {
    use std::ffi::{c_char, c_void};
    type CfType = *const c_void;
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: CfType,
            allocator: CfType,
            options: u32,
        ) -> CfType;
        fn IOObjectRelease(object: u32) -> i32;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: CfType,
            text: *const c_char,
            encoding: u32,
        ) -> CfType;
        fn CFGetTypeID(object: CfType) -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFNumberGetValue(number: CfType, kind: isize, value: *mut c_void) -> u8;
        fn CFRelease(object: CfType);
    }
    const UTF8: u32 = 0x0800_0100; // kCFStringEncodingUTF8
    const SINT64: isize = 4; // kCFNumberSInt64Type
    const MAIN_PORT: u32 = 0; // kIOMainPortDefault
    let null = std::ptr::null();
    // SAFETY: the class name is NUL-terminated; IOServiceGetMatchingService
    // consumes the matching dictionary (a null one is checked first).
    let service = unsafe {
        let matching = IOServiceMatching(c"AGXAccelerator".as_ptr());
        if matching.is_null() {
            return None;
        }
        IOServiceGetMatchingService(MAIN_PORT, matching)
    };
    if service == 0 {
        return None;
    }
    // SAFETY: `key` is a CFString created here (or null, checked) and released
    // once; `service` is a live registry entry released once; the property is
    // only read as a CFNumber after its type is checked, then released once.
    // Reading a registry property changes nothing.
    unsafe {
        let key = CFStringCreateWithCString(null, c"gpu-core-count".as_ptr(), UTF8);
        let value = if key.is_null() {
            null
        } else {
            IORegistryEntryCreateCFProperty(service, key, null, 0)
        };
        if !key.is_null() {
            CFRelease(key);
        }
        IOObjectRelease(service);
        if value.is_null() {
            return None;
        }
        let mut count = 0_i64;
        let read = CFGetTypeID(value) == CFNumberGetTypeID()
            && CFNumberGetValue(value, SINT64, (&raw mut count).cast()) != 0;
        CFRelease(value);
        read.then(|| usize::try_from(count).ok())
            .flatten()
            .filter(|&n| n > 0)
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_core_count() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threads_follow_core_levels_with_fallbacks_for_missing_names() {
        let machine = |levels: &[usize], physical, available| Machine {
            core_levels: levels.to_vec(),
            physical_cores: physical,
            available_cpus: available,
            ..Machine::default()
        };
        // The measured M5: 4 performance + 6 efficiency cores.
        assert_eq!(machine(&[4, 6], Some(10), Some(10)).default_threads(), 10);
        assert_eq!(machine(&[12, 4], Some(16), Some(16)).default_threads(), 16);
        assert_eq!(machine(&[24, 8], Some(32), Some(32)).default_threads(), 32);
        // A process limited to fewer CPUs never gets more workers.
        assert_eq!(machine(&[12, 4], Some(16), Some(6)).default_threads(), 6);
        // Missing hw.perflevel names: physical cores, then available CPUs, then 1.
        assert_eq!(machine(&[], Some(8), Some(8)).default_threads(), 8);
        assert_eq!(machine(&[], None, Some(12)).default_threads(), 12);
        assert_eq!(machine(&[], None, None).default_threads(), 1);
        // Zero counts are ignored like missing ones.
        assert_eq!(machine(&[0, 0], Some(8), Some(8)).default_threads(), 8);
        assert_eq!(machine(&[], Some(0), Some(6)).default_threads(), 6);
        assert_eq!(machine(&[], Some(0), Some(0)).default_threads(), 1);
        assert_eq!(machine(&[0], None, None).default_threads(), 1);
        // A sum that overflows falls back; absurd counts are bounded by the
        // CPUs available, or by the constant when those are unknown.
        let max = Machine::MAX_UNBOUNDED_THREADS;
        assert_eq!(
            machine(&[usize::MAX, 1], Some(10), Some(10)).default_threads(),
            10
        );
        assert_eq!(machine(&[usize::MAX, 1], None, None).default_threads(), 1);
        assert_eq!(
            machine(&[1 << 40, 4], Some(16), Some(12)).default_threads(),
            12
        );
        assert_eq!(machine(&[1 << 40, 4], None, None).default_threads(), max);
        assert_eq!(machine(&[], Some(usize::MAX), None).default_threads(), max);
        assert_eq!(machine(&[24, 8], Some(32), None).default_threads(), 32);
    }

    #[test]
    fn this_machine_reports_consistent_facts() {
        let machine = Machine::detect();
        println!("{machine:?}");
        assert!(machine.default_threads() >= 1);
        if cfg!(target_os = "macos") {
            assert!(machine.memory_bytes.is_some_and(|n| n > 0));
            if let (false, Some(physical)) =
                (machine.core_levels.is_empty(), machine.physical_cores)
            {
                assert_eq!(machine.core_levels.iter().sum::<usize>(), physical);
            }
        }
        assert!(machine.gpu_cores.is_none_or(|n| (1..1024).contains(&n)));
        assert_eq!(sysctl_u64("mackani.no.such.name"), None);
        assert_eq!(sysctl_string("mackani.no.such.name"), None);
    }
}
