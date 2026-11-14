//! Real-model losslessness check through the product request and chat path.
mod lossless_support;

fn main() -> std::process::ExitCode {
    let result = lossless_support::run(std::env::args().skip(1), |config| {
        Ok(Box::new(mackani::real_engine::RealEngine::load(config)?))
    });
    match result {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("lossless: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
