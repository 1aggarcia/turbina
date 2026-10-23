use ureq::{Agent, Error, RequestBuilder};

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
        HttpRequest::Delete(r) => {
            attach_headers(agent.delete(r.url), r.headers).call()?
        }
        HttpRequest::Get(r) => {
            attach_headers(agent.get(r.url), r.headers).call()?
        }
        HttpRequest::Patch(r) => {
            attach_headers(agent.patch(r.url), r.headers).send(r.body)?
        }
        HttpRequest::Post(r) => {
            attach_headers(agent.post(r.url), r.headers).send(r.body)?
        }
        HttpRequest::Put(r) => {
            attach_headers(agent.put(r.url), r.headers).send(r.body)?
        }
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

fn attach_headers<T>(
    request_builder: RequestBuilder<T>,
    headers: Vec<String>,
) -> RequestBuilder<T> {
    headers.iter().fold(request_builder, |builder, header| {
        let key_value_pair: Vec<_> =
            header.split(":").into_iter().map(|s| s.trim()).collect();

        match (key_value_pair.get(0), key_value_pair.get(1)) {
            (Some(key), Some(value)) => builder.header(*key, *value),
            _ => builder,
        }
    })
}

#[cfg(test)]
mod test {
    use super::*;

    mod test_attach_headers {
        use super::*;

        #[test]
        fn it_should_parse_and_attach_headers() {
            let builder = ureq::get("http://httpbin.org/get");
            let input_headers: Vec<String> = vec![
                "Content-Type: application/json".into(),
                "Test2 : space on both sides".into(),
                "invalid - should ignore".into(),
                "Test3:no-spaces".into(),
            ];
            let output = attach_headers(builder, input_headers);
            let output_headers = output.headers_ref().unwrap();

            assert_eq!(output_headers["Content-Type"], "application/json");
            assert_eq!(output_headers["Test2"], "space on both sides");
            assert_eq!(output_headers["Test3"], "no-spaces");
            assert_eq!(output_headers.keys_len(), 3);
        }

        #[test]
        fn it_should_handle_empty_header_list() {
            let builder = ureq::get("http://httpbin.org/get");
            let output = attach_headers(builder, vec![]);
            let output_headers = output.headers_ref().unwrap();
            assert_eq!(output_headers.keys_len(), 0);
        }
    }
}
