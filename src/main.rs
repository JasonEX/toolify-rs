use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use futures_util::future;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
#[cfg(unix)]
use socket2::{Domain, Protocol, Socket, Type};
use toolify_rs::auth::build_allowed_key_set;
use toolify_rs::config::{load_config, AppConfig, ServerConfig};
use toolify_rs::observability::init_tracing;
use toolify_rs::routing::dispatch::{dispatch_request, normalize_base_path};
use toolify_rs::routing::ModelRouter;
use toolify_rs::state::AppState;
use toolify_rs::transport::{HttpTransport, PreparedUpstream};

const DEFAULT_LISTEN_BACKLOG: i32 = 1024;

fn main() {
    let config = load_config("config.yaml").unwrap_or_else(|e| {
        eprintln!("Failed to load configuration: {e}");
        eprintln!("Please copy 'config.example.yaml' to 'config.yaml' and modify as needed.");
        std::process::exit(1);
    });

    init_tracing(&config.features.log_level);
    let runtime = build_runtime(&config);

    runtime.block_on(async move {
        run(config).await;
    });
}

fn build_runtime(config: &AppConfig) -> tokio::runtime::Runtime {
    let worker_threads = config.server.runtime_worker_threads;
    let max_blocking_threads = config.server.runtime_max_blocking_threads;
    let thread_stack_size_kb = config.server.runtime_thread_stack_size_kb;
    let mut runtime_builder = if worker_threads == Some(1) {
        tokio::runtime::Builder::new_current_thread()
    } else {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        if let Some(threads) = worker_threads {
            builder.worker_threads(threads);
        }
        builder
    };
    runtime_builder.enable_io();
    runtime_builder.enable_time();
    if let Some(max_blocking_threads) = max_blocking_threads {
        runtime_builder.max_blocking_threads(max_blocking_threads);
    }
    if worker_threads != Some(1) {
        if let Some(thread_stack_size_kb) = thread_stack_size_kb {
            runtime_builder.thread_stack_size(thread_stack_size_kb * 1024);
        }
    }
    runtime_builder.build().unwrap_or_else(|e| {
        eprintln!("Failed to initialize Tokio runtime: {e}");
        std::process::exit(1);
    })
}

async fn run(config: AppConfig) {
    let host = config.server.host.clone();
    let port = config.server.port;
    let base_path = normalize_base_path(&config.server.base_path);

    let model_router = ModelRouter::new(&config);
    let prepared_upstreams = config
        .upstream_services
        .iter()
        .map(PreparedUpstream::new)
        .collect();
    let allowed_client_keys = build_allowed_key_set(&config);
    let transport = HttpTransport::new_with_upstream_count_and_proxies(
        &config.server,
        config.upstream_services.len(),
        config
            .upstream_services
            .iter()
            .flat_map(|upstream| {
                [
                    upstream.proxy.as_deref(),
                    upstream.proxy_stream.as_deref(),
                    upstream.proxy_non_stream.as_deref(),
                ]
            })
            .flatten(),
    );
    let state = Arc::new(AppState::new(
        config,
        transport,
        model_router,
        prepared_upstreams,
        allowed_client_keys,
    ));
    let dispatch_state = Arc::clone(&state);
    let dispatch_base_path = Arc::<str>::from(base_path.clone());

    tracing::info!(
        "toolify-rs starting on {}:{} with base_path='{}'",
        host,
        port,
        base_path
    );

    let listeners = build_server_listeners(&state.config.server, &host, port)
        .await
        .unwrap_or_else(|err| {
            eprintln!("Failed to bind to {host}:{port}: {err}");
            std::process::exit(1);
        });
    let reuse_port_enabled = state.config.server.tcp_reuse_port_listener_count.is_some();

    tracing::info!(
        "toolify-rs is ready to accept connections (listeners={}, reuse_port={})",
        listeners.len(),
        reuse_port_enabled
    );
    let conn_builder = AutoBuilder::new(TokioExecutor::new());
    if listeners.len() == 1 {
        let mut listeners = listeners;
        let Some(listener) = listeners.pop() else {
            return;
        };
        serve_accept_loop(
            listener,
            conn_builder,
            Arc::clone(&dispatch_state),
            Arc::clone(&dispatch_base_path),
        )
        .await;
        return;
    }

    for listener in listeners {
        let loop_builder = conn_builder.clone();
        let request_state = Arc::clone(&dispatch_state);
        let request_base_path = Arc::clone(&dispatch_base_path);
        tokio::spawn(async move {
            serve_accept_loop(listener, loop_builder, request_state, request_base_path).await;
        });
    }
    future::pending::<()>().await;
}

async fn serve_accept_loop(
    listener: tokio::net::TcpListener,
    conn_builder: AutoBuilder<TokioExecutor>,
    dispatch_state: Arc<AppState>,
    dispatch_base_path: Arc<str>,
) {
    loop {
        let (stream, remote_addr) = match listener.accept().await {
            Ok((stream, remote_addr)) => (stream, remote_addr),
            Err(err) => {
                eprintln!("Accept error: {err}");
                continue;
            }
        };

        if let Err(err) = stream.set_nodelay(true) {
            tracing::debug!("failed to enable TCP_NODELAY for {remote_addr}: {err}");
        }

        let io = TokioIo::new(stream);
        let conn_builder = conn_builder.clone();
        let request_state = Arc::clone(&dispatch_state);
        let request_base_path = Arc::clone(&dispatch_base_path);
        let hyper_service = service_fn(move |request: Request<Incoming>| {
            dispatch_request(
                Arc::clone(&request_state),
                Arc::clone(&request_base_path),
                request.map(Body::new),
            )
        });

        tokio::spawn(async move {
            if let Err(err) = conn_builder.serve_connection(io, hyper_service).await {
                tracing::debug!("failed to serve connection from {remote_addr}: {err:#}");
            }
        });
    }
}

async fn build_server_listeners(
    server: &ServerConfig,
    host: &str,
    port: u16,
) -> io::Result<Vec<tokio::net::TcpListener>> {
    let reuse_port_enabled = server.tcp_reuse_port_listener_count.is_some();
    let mut listener_count = reuse_port_listener_count(server);
    if !reuse_port_enabled {
        listener_count = 1;
    }
    if reuse_port_enabled && !reuse_port_supported() {
        tracing::warn!(
            "server.tcp_reuse_port_listener_count is set but this platform does not support SO_REUSEPORT; fallback to single listener"
        );
        listener_count = 1;
    }

    if listener_count == 1 {
        let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
        return Ok(vec![listener]);
    }

    let mut listeners = Vec::with_capacity(listener_count);
    for _ in 0..listener_count {
        listeners.push(bind_reuse_port_listener(host, port)?);
    }
    Ok(listeners)
}

#[must_use]
fn reuse_port_listener_count(server: &ServerConfig) -> usize {
    if let Some(explicit) = server.tcp_reuse_port_listener_count {
        return explicit.max(1);
    }
    1
}

#[must_use]
fn reuse_port_supported() -> bool {
    cfg!(unix)
}

#[cfg(unix)]
fn bind_reuse_port_listener(host: &str, port: u16) -> io::Result<tokio::net::TcpListener> {
    let mut last_err = None;
    for addr in (host, port).to_socket_addrs()? {
        match bind_reuse_port_listener_addr(addr) {
            Ok(listener) => return Ok(listener),
            Err(err) => last_err = Some(err),
        }
    }

    Err(last_err.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AddrNotAvailable,
            format!("no bindable socket address for {host}:{port}"),
        )
    }))
}

#[cfg(unix)]
fn bind_reuse_port_listener_addr(addr: SocketAddr) -> io::Result<tokio::net::TcpListener> {
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.bind(&addr.into())?;
    socket.listen(DEFAULT_LISTEN_BACKLOG)?;
    socket.set_nonblocking(true)?;

    let std_listener: std::net::TcpListener = socket.into();
    tokio::net::TcpListener::from_std(std_listener)
}

#[cfg(not(unix))]
fn bind_reuse_port_listener(_host: &str, _port: u16) -> io::Result<tokio::net::TcpListener> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "SO_REUSEPORT is only supported on Unix-like platforms",
    ))
}
