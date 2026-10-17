#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // WASM doesn't read the enum fields
pub enum HttpRequest {
    Delete { url: String },
    Get { url: String },
    Post { url: String, body: String },
    Put { url: String, body: String },
    Patch { url: String, body: String },
}

pub struct HttpResponse {
    pub status_code: i32,
    pub body: String,
    pub headers: Vec<String>,
}
