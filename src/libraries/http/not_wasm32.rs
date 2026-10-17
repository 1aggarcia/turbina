use ureq::{Agent, Error};

use crate::libraries::http::shared::{HttpRequest, HttpResponse};

pub fn make_http_request(request: HttpRequest) -> Result<HttpResponse, String> {
    // Convert ureq Error type to String since ureq Errors can't compile on WASM
    make_http_request_with_ureq_error(request).map_err(|err| err.to_string())
}

fn make_http_request_with_ureq_error(
    request: HttpRequest,
) -> Result<HttpResponse, Error> {
    // Needs separate agent with config to treat 4XX/5XX statuses as success
    let agent: Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    let mut response = match request {
        HttpRequest::Delete { url } => agent.delete(url).call()?,
        HttpRequest::Get { url } => agent.get(url).call()?,
        HttpRequest::Patch { url, body } => agent.patch(url).send(body)?,
        HttpRequest::Post { url, body } => agent.post(url).send(body)?,
        HttpRequest::Put { url, body } => agent.put(url).send(body)?,
    };

    let status_code = i32::from(response.status().as_u16());
    let body = response.body_mut().read_to_string()?;

    let headers: Vec<String> = response
        .headers()
        .iter()
        .map(|(name, value)| {
            let str_name = name.to_string();
            let str_value = value.to_str().map(|s| s.to_owned()).unwrap_or(
                // Should be string most of the time, but the HTTP spec
                // allows opaque bytes as well, so we fall back to that
                value
                    .as_bytes()
                    .iter()
                    .map(|b| format!("{:02X}", b))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            format!("{}: {}", str_name, str_value)
        })
        .collect();

    Ok(HttpResponse {
        status_code,
        body,
        headers,
    })
}
