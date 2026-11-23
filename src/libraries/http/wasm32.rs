use crate::libraries::http::shared::HttpRequest;
use crate::libraries::TurbinaHttpResponse;

pub fn make_http_request(
    _request: HttpRequest,
) -> Result<TurbinaHttpResponse, String> {
    Err("HTTP requests not supported in the online playground".into())
}
