use once_cell::sync::Lazy;
use ureq::{Agent, Error};

use crate::{libraries::factories::{create_result_from_error, create_result_from_success}, models::{EvalContext, Literal, Type}};

enum HttpRequest {
    Get { url: String },
    // add more methods here
}

struct HttpResponse {
    status_code: i32,
    body: String,
    headers: Vec<String>,
}

pub static HTTP_REQUEST_SUCCESS_TYPES: Lazy<[Type; 3]> = Lazy::new(|| [
    Type::Int, // status code
    Type::String, // response body
    Type::String.as_list(), // response headers
    // TODO: change to struct once struct literals are supported
    /*
    Type::Struct(HashMap::from([
        ("statusCode".into(), Type::Int),
        ("body".into(), Type::String),
        ("headers".into(), Type::String.as_list()),
    ]))
    */
]);

pub fn lib_http_get(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
    }
    handle_http_request(HttpRequest::Get { url: url.to_string() }, context)
}

fn handle_http_request(method: HttpRequest, context: &mut EvalContext) -> Literal {
    match make_http_request(method) {
        Ok(response) => {
            // TODO: change to struct once struct literals are supported
            let status_literal = Literal::Int(response.status_code);
            let body_literal = Literal::String(response.body);
            let headers_literal = Literal::List(
                response.headers
                    .into_iter()
                    .map(|h| Literal::String(h))
                    .collect()
            );
            create_result_from_success(
                vec![
                    status_literal,
                    body_literal,
                    headers_literal
                ],
                &*HTTP_REQUEST_SUCCESS_TYPES,
                context
            )
        },
        Err(error) => {
            create_result_from_error(
                Literal::String(error.to_string()),
                &*HTTP_REQUEST_SUCCESS_TYPES,
                context
            )
        }
    }
}

fn make_http_request(request: HttpRequest) -> Result<HttpResponse, Error> {
    // Needs separate agent with config to treat 4XX/5XX statuses as success
    let agent: Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();

    let mut response = match request {
        HttpRequest::Get { url } => agent.get(url).call()?,
    };

    let status_code = i32::from(response.status().as_u16());
    let body = response
        .body_mut()
        .read_to_string()?;

    let headers: Vec<String> = response.headers()
        .iter()
        .map(|(name, value)| {
            let str_name = name.to_string();
            let str_value = value.to_str()
                .map(|s| s.to_owned())
                .unwrap_or(
                    // Should be string most of the time, but the HTTP spec
                    // allows opaque bytes as well, so we fall back to that
                    value
                        .as_bytes()
                        .iter()
                        .map(|b| format!("{:02X}", b))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            format!("{}: {}", str_name, str_value)
        })
        .collect();

    Ok(HttpResponse { status_code, body, headers })
}
