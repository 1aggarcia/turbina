use crate::libraries::http::shared::{HttpRequest, HttpResponse};

pub fn make_http_request(
    _request: HttpRequest,
) -> Result<HttpResponse, String> {
    Err("HTTP requests not supported in the online playground".into())
}
