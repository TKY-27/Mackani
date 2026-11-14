use super::{Api, Config, Limits};
use crate::{Error, Result, chat::ChatRunner};
use hyper::{Request, server::conn::http1, service::service_fn};
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    convert::Infallible,
    future::Future,
    io,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, ready},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::Semaphore,
    task::JoinSet,
    time::{Instant, Sleep},
};

/// A connection's "the server works on a request" flag, shared between its
/// `IdleIo` and the request it serves (as an extension). Set from the end of
/// the request body until the response is complete: queued, prepared,
/// generating, streaming. Requests served by an embedder carry none.
#[derive(Clone, Default)]
pub(super) struct Working(pub(super) Arc<AtomicBool>);
impl Working {
    pub(super) fn start(&self) -> WorkingGuard {
        self.0.store(true, Ordering::Release);
        WorkingGuard(self.0.clone())
    }
}
pub(super) struct WorkingGuard(pub(super) Arc<AtomicBool>);
impl Drop for WorkingGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
pub struct Server {
    listener: TcpListener,
    api: Arc<Api>,
}
impl Server {
    pub async fn bind<F>(mut cfg: Config, factory: F) -> Result<Self>
    where
        F: FnOnce() -> Result<ChatRunner> + Send + 'static,
    {
        cfg.validate()?;
        let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
            .await
            .map_err(|e| Error::engine(format!(
                "cannot listen on {}:{} ({e}). Is another server already using this port? Use --port to pick another.",
                cfg.host, cfg.port
            )))?;
        cfg.port = listener.local_addr().map_err(Error::engine)?.port();
        Ok(Self {
            listener,
            api: Arc::new(Api::new(cfg, factory)?),
        })
    }
    pub fn local_addr(&self) -> Result<std::net::SocketAddr> {
        self.listener.local_addr().map_err(Error::engine)
    }
    pub fn is_ready(&self) -> bool {
        self.api.is_ready()
    }
    pub async fn run(self, shutdown: impl Future<Output = ()>) -> Result<()> {
        tokio::pin!(shutdown);
        let permits = Arc::new(Semaphore::new(self.api.shared.cfg.limits.max_connections));
        let mut connections = JoinSet::new();
        let result = loop {
            tokio::select! {
                _ = &mut shutdown => break Ok(()),
                _ = self.api.shared.failure_ready.notified() => {
                    let failure = self.api.shared.failure.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    if let Some(error) = failure {
                        break Err(error);
                    }
                }
                Some(_) = connections.join_next(), if !connections.is_empty() => {}
                accepted = self.listener.accept() => {
                    let (socket, _) = match accepted {
                        Ok(connection) => connection,
                        Err(error) => break Err(Error::engine(error)),
                    };
                    let Ok(permit) = permits.clone().try_acquire_owned() else {
                        drop(socket);
                        continue;
                    };
                    let _ = socket.set_nodelay(true);
                    let api = self.api.clone();
                    connections.spawn(async move {
                        let _permit = permit;
                        serve_connection(socket, api).await;
                    });
                }
            }
        };
        self.api.shared.shutdown.store(true, Ordering::Release);
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        drop(self.api);
        result
    }
}

async fn serve_connection(socket: TcpStream, api: Arc<Api>) {
    let limits = &api.shared.cfg.limits;
    let working = Working::default();
    let io = TokioIo::new(IdleIo::new(socket, limits, working.0.clone()));
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(limits.header_timeout)
        .max_headers(limits.max_headers)
        .max_buf_size(limits.header_bytes);
    let service = service_fn(move |mut request: Request<hyper::body::Incoming>| {
        request.extensions_mut().insert(working.clone());
        let api = api.clone();
        async move { Ok::<_, Infallible>(api.call(request).await) }
    });
    let _ = builder.serve_connection(io, service).await;
}

// The idle deadline bounds every wait for the client: reading a request head
// or body, the next request on a persistent connection, and a write that makes
// no progress (a peer that stops reading). While the connection's request is
// worked on (`Working`), Hyper still reads, but only to notice a disconnect,
// which cancels the engine; those reads have no deadline, so a long silent
// prefill is not cut off. A boxed Sleep keeps pinning safe.
struct IdleIo {
    socket: TcpStream,
    sleep: Pin<Box<Sleep>>,
    timeout: Duration,
    working: Arc<AtomicBool>,
    // `working` as last seen by a read: a change restarts the read deadline.
    was_working: bool,
    linger_bytes: usize,
    linger_timeout: Duration,
    // Some once shutdown has half-closed: the bytes still allowed to discard.
    linger_left: Option<usize>,
}
impl IdleIo {
    fn new(socket: TcpStream, limits: &Limits, working: Arc<AtomicBool>) -> Self {
        Self {
            socket,
            sleep: Box::pin(tokio::time::sleep(limits.idle_timeout)),
            timeout: limits.idle_timeout,
            working,
            was_working: false,
            linger_bytes: limits.linger_bytes,
            linger_timeout: limits.linger_timeout,
            linger_left: None,
        }
    }
    fn expired(&mut self, cx: &mut Context<'_>) -> bool {
        self.sleep.as_mut().poll(cx).is_ready()
    }
    fn progress(&mut self) {
        self.sleep.as_mut().reset(Instant::now() + self.timeout);
    }
}
impl AsyncRead for IdleIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let working = this.working.load(Ordering::Acquire);
        if working != this.was_working {
            // A response just completed (or work began): the wait starts now,
            // not at the last bytes before a possibly long silence.
            this.was_working = working;
            this.progress();
        }
        if !working && this.expired(cx) {
            return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
        }
        let before = buf.filled().len();
        let r = Pin::new(&mut this.socket).poll_read(cx, buf);
        if buf.filled().len() > before {
            this.progress();
        }
        r
    }
}
impl AsyncWrite for IdleIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let r = Pin::new(&mut this.socket).poll_write(cx, buf);
        match &r {
            Poll::Ready(Ok(n)) if *n > 0 => this.progress(),
            // Only a write that waits for the peer is held to the deadline: the
            // first output after a long silence is not refused for the silence.
            Poll::Pending if this.expired(cx) => {
                return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
            }
            _ => {}
        }
        r
    }
    // A TCP flush never waits, so it has no deadline of its own.
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().socket).poll_flush(cx)
    }
    // Hyper calls this after flushing every response it ends a connection with,
    // including its own 400/431 parse-error responses. Closing a socket with
    // unread received bytes makes the kernel send RST, which can destroy the
    // response before the peer reads it; so half-close, then drain until the
    // peer's EOF, a byte budget or a deadline, whichever comes first.
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.linger_left.is_none() {
            ready!(Pin::new(&mut this.socket).poll_shutdown(cx))?;
            this.sleep
                .as_mut()
                .reset(Instant::now() + this.linger_timeout);
            this.linger_left = Some(this.linger_bytes);
        }
        let mut scratch = [0_u8; 4096];
        while let Some(left) = this.linger_left.filter(|&n| n > 0) {
            if this.expired(cx) {
                break;
            }
            let mut buf = ReadBuf::new(&mut scratch[..left.min(4096)]);
            match Pin::new(&mut this.socket).poll_read(cx, &mut buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(())) if !buf.filled().is_empty() => {
                    this.linger_left = Some(left - buf.filled().len());
                }
                // EOF or a reset: nothing is left to protect.
                Poll::Ready(_) => break,
            }
        }
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    // Hyper 1.12 writes a completed response before it reads again, so the
    // restart of the read deadline when work ends is checked here directly.
    #[tokio::test]
    async fn reads_have_no_deadline_while_working_and_a_fresh_one_after() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let idle = Duration::from_millis(300);
        let limits = Limits {
            idle_timeout: idle,
            ..Limits::default()
        };
        let working = Arc::new(AtomicBool::new(true));
        let mut io = IdleIo::new(socket, &limits, working.clone());
        let mut byte = [0_u8; 1];
        // Three idle deadlines of silence while the request is worked on.
        let silent = tokio::time::timeout(3 * idle, io.read(&mut byte)).await;
        assert!(silent.is_err(), "{silent:?}");
        working.store(false, Ordering::Release);
        let start = Instant::now();
        let r = tokio::time::timeout(Duration::from_secs(30), io.read(&mut byte))
            .await
            .expect("the idle deadline never fired");
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(start.elapsed() >= idle, "{:?}", start.elapsed());
        drop(client);
    }
}
