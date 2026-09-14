use once_cell::sync::Lazy;
use ureq::{Agent, Error};

use crate::{libraries::factories::{create_result_from_error, create_result_from_success, create_result_type}, models::{EvalContext, FuncBody, Function, Literal, Type}};

pub enum HttpBodySetting {
    NoBody,
    RequireBody
}

enum HttpRequest {
    Delete { url: String },
    Get { url: String },
    Post { url: String, body: String },
    Put { url: String, body: String },
    Patch { url: String, body: String },
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

/// Factory to help create function AST nodes for different HTTP methods,
/// as they are all mostly the same and only have different symbols for
/// user convenience.
pub fn create_http_function(
    http_handler: fn(Vec<Literal>, &mut EvalContext) -> Literal,
    body_setting: HttpBodySetting
) -> Function {
    let params = match body_setting {
        HttpBodySetting::NoBody => define_params![
            url = Type::String,
        ],
        HttpBodySetting::RequireBody => define_params![
            url = Type::String,
            body = Type::String,
        ]
    };
    Function {
        type_params: vec![],
        params,
        return_type: Some(create_result_type(&*HTTP_REQUEST_SUCCESS_TYPES)),
        body: FuncBody::Native(http_handler)
    }
}

pub fn lib_http_delete(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args => Literal::String(url) }
    handle_http_request(HttpRequest::Delete { url: url.to_string() }, context)
}

pub fn lib_http_get(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args => Literal::String(url) }
    handle_http_request(HttpRequest::Get { url: url.to_string() }, context)
}

pub fn lib_http_patch(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(HttpRequest::Patch {
        url: url.to_string(),
        body: request_body.to_string(),
    }, context)
}

pub fn lib_http_post(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(HttpRequest::Post {
        url: url.to_string(),
        body: request_body.to_string(),
    }, context)
}

pub fn lib_http_put(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(HttpRequest::Put {
        url: url.to_string(),
        body: request_body.to_string(),
    }, context)
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
        HttpRequest::Delete { url } => agent.delete(url).call()?,
        HttpRequest::Get { url } => agent.get(url).call()?,
        HttpRequest::Patch { url, body } => agent.patch(url).send(body)?,
        HttpRequest::Post { url, body } => agent.post(url).send(body)?,
        HttpRequest::Put { url, body } => agent.put(url).send(body)?,
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
