use std::error::Error;

use once_cell::sync::Lazy;

mod shared;

#[cfg(target_arch = "wasm32")]
mod wasm32;

#[cfg(target_arch = "wasm32")]
use crate::libraries::http::wasm32::make_http_request;

#[cfg(not(target_arch = "wasm32"))]
mod not_wasm32;

#[cfg(not(target_arch = "wasm32"))]
use crate::libraries::http::not_wasm32::make_http_request;

use crate::{
    libraries::{
        factories::{
            create_result_from_error, create_result_from_success,
            create_result_type,
        },
        http::shared::{HttpRequest, RequestWithBody, RequestWithoutBody},
    },
    models::{EvalContext, FuncBody, Function, Literal, Type},
};

pub enum HttpBodySetting {
    NoBody,
    RequireBody,
}

pub static HTTP_REQUEST_SUCCESS_TYPES: Lazy<[Type; 3]> = Lazy::new(|| {
    [
        Type::Int,    // status code
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
    ]
});

/// Factory to help create function AST nodes for different HTTP methods,
/// as they are all mostly the same and only have different symbols for
/// user convenience.
pub fn create_http_function(
    http_handler: fn(Vec<Literal>, &mut EvalContext) -> Literal,
    body_setting: HttpBodySetting,
) -> Function {
    let params: Vec<(String, Type)> = match body_setting {
        HttpBodySetting::NoBody => {
            define_params![url = Type::String, headers = Type::String.as_list()]
        }
        HttpBodySetting::RequireBody => {
            define_params![
                url = Type::String,
                headers = Type::String.as_list(),
                body = Type::String
            ]
        }
    };
    Function {
        type_params: vec![],
        params,
        return_type: Some(create_result_type(&*HTTP_REQUEST_SUCCESS_TYPES)),
        body: FuncBody::Native(http_handler),
    }
}

pub fn lib_http_delete(
    args: Vec<Literal>,
    context: &mut EvalContext,
) -> Literal {
    let request = get_request_without_body(args);
    handle_http_request(HttpRequest::Delete(request), context)
}

pub fn lib_http_get(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let request = get_request_without_body(args);
    handle_http_request(HttpRequest::Get(request), context)
}

pub fn lib_http_patch(
    args: Vec<Literal>,
    context: &mut EvalContext,
) -> Literal {
    let request = get_request_with_body(args);
    handle_http_request(HttpRequest::Patch(request), context)
}

pub fn lib_http_post(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let request = get_request_with_body(args);
    handle_http_request(HttpRequest::Post(request), context)
}

pub fn lib_http_put(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let request = get_request_with_body(args);
    handle_http_request(HttpRequest::Put(request), context)
}

// TODO: split client and server into separate files
pub fn open_http_server<F>(
    address: &str,
    mut handle_request: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut tiny_http::Request) -> String,
{
    let server = tiny_http::Server::http(address).map_err(|e| e.to_string())?;

    /*
     * The Rustyline editor overrides the Ctrl+C handler, meaning
     * Ctrl+C does not terminate the program while this blocking
     * server is running in the REPL. I could not find a clean way to
     * get around this, so I just reset the signal handlers back to
     * system defaults manually with libc (as this function never
     * returns anyways).
     *
     * https://github.com/kkawakam/rustyline/issues/879
     */
    #[cfg(not(target_arch = "wasm32"))]
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }
    println!("Server listening on address: {}", address);

    for mut request in server.incoming_requests() {
        let response_text = handle_request(&mut request);
        let response = tiny_http::Response::from_string(response_text);
        request.respond(response)?;
    }
    Ok(())
}

fn get_request_with_body(args: Vec<Literal>) -> RequestWithBody {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::List(headers),
        Literal::String(body),
    };
    RequestWithBody {
        url: url.to_string(),
        headers: parse_headers(headers),
        body: body.to_string(),
    }
}

fn get_request_without_body(args: Vec<Literal>) -> RequestWithoutBody {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::List(headers),
    };
    RequestWithoutBody {
        url: url.to_string(),
        headers: parse_headers(headers),
    }
}

/// Convert list of literals from Turbina to a list of native strings
/// for processing
fn parse_headers(list: &Vec<Literal>) -> Vec<String> {
    list.into_iter()
        .map(|item| match item {
            Literal::String(string_item) => string_item.to_string(),
            _ => panic!(
                "Unexpected argument type(s) passed to function: {:?}",
                item
            ),
        })
        .collect()
}

fn handle_http_request(
    method: HttpRequest,
    context: &mut EvalContext,
) -> Literal {
    match make_http_request(method) {
        Ok(response) => {
            // TODO: change to struct once struct literals are supported
            let status_literal = Literal::Int(response.status_code);
            let body_literal = Literal::String(response.body);
            let headers_literal = Literal::List(
                response
                    .headers
                    .into_iter()
                    .map(|h| Literal::String(h))
                    .collect(),
            );
            create_result_from_success(
                vec![status_literal, body_literal, headers_literal],
                &*HTTP_REQUEST_SUCCESS_TYPES,
                context,
            )
        }
        Err(error) => create_result_from_error(
            Literal::String(error.to_string()),
            &*HTTP_REQUEST_SUCCESS_TYPES,
            context,
        ),
    }
}
