#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // WASM doesn't read the enum fields
pub enum HttpRequest {
    Delete(RequestWithoutBody),
    Get(RequestWithoutBody),
    Post(RequestWithBody),
    Put(RequestWithBody),
    Patch(RequestWithBody),
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // WASM doesn't read this struct
pub struct RequestWithBody {
    pub url: String,
    pub headers: Vec<String>,
    pub body: String,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // WASM doesn't read this struct
pub struct RequestWithoutBody {
    pub url: String,
    pub headers: Vec<String>,
}

pub struct HttpResponse {
    pub status_code: i32,
    pub body: String,
    pub headers: Vec<String>,
}
