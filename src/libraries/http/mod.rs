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
        http::shared::HttpRequest,
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
    let params = match body_setting {
        HttpBodySetting::NoBody => define_params![url = Type::String,],
        HttpBodySetting::RequireBody => {
            define_params![url = Type::String, body = Type::String,]
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
    unwrap_args! { args => Literal::String(url) }
    handle_http_request(
        HttpRequest::Delete {
            url: url.to_string(),
        },
        context,
    )
}

pub fn lib_http_get(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args => Literal::String(url) }
    handle_http_request(
        HttpRequest::Get {
            url: url.to_string(),
        },
        context,
    )
}

pub fn lib_http_patch(
    args: Vec<Literal>,
    context: &mut EvalContext,
) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(
        HttpRequest::Patch {
            url: url.to_string(),
            body: request_body.to_string(),
        },
        context,
    )
}

pub fn lib_http_post(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(
        HttpRequest::Post {
            url: url.to_string(),
            body: request_body.to_string(),
        },
        context,
    )
}

pub fn lib_http_put(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    unwrap_args! { args =>
        Literal::String(url),
        Literal::String(request_body),
    }
    handle_http_request(
        HttpRequest::Put {
            url: url.to_string(),
            body: request_body.to_string(),
        },
        context,
    )
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
