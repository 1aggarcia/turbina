//! Global structs accessible anywhere in Turbina. Typically used for standard
//! library functions. Uses static definitions so that the definitions are
//! never copied at runtime.

use std::sync::LazyLock;

use crate::{
    factories::field_map,
    models::{Literal, StructDefinition, StructLiteral, Type},
};

pub static GLOBAL_STRUCTS: LazyLock<Vec<&'static StructDefinition>> =
    LazyLock::new(|| {
        vec![
            TurbinaHttpRequest::get_definition(),
            TurbinaHttpResponse::get_definition(),
        ]
    });

/// Struct that can be translated into a Turbina struct for usage inside a
/// Turbina program
pub trait TurbinaStruct {
    fn get_definition() -> &'static StructDefinition;

    fn get_type() -> Type;

    fn to_literal(self) -> Literal;
}

// ---------- Individual Struct definitions ---------- //

pub struct TurbinaHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<String>,
    pub body: String,
}

static HTTP_REQUEST_DEFINITION: LazyLock<StructDefinition> =
    LazyLock::new(|| StructDefinition {
        name: "HttpRequest".to_string(),
        fields: field_map([
            ("method", Type::String),
            ("url", Type::String),
            ("headers", Type::String.as_list()),
            ("body", Type::String),
        ]),
    });

impl TurbinaStruct for TurbinaHttpRequest {
    fn get_definition() -> &'static StructDefinition {
        &*HTTP_REQUEST_DEFINITION
    }

    fn get_type() -> Type {
        Type::StructV2(HTTP_REQUEST_DEFINITION.name.to_string())
    }

    fn to_literal(self) -> Literal {
        let headers = self
            .headers
            .into_iter()
            .map(|h| Literal::String(h))
            .collect();

        Literal::Struct(StructLiteral {
            name: HTTP_REQUEST_DEFINITION.name.to_string(),
            fields: field_map([
                ("method", Literal::String(self.method)),
                ("url", Literal::String(self.url)),
                ("headers", Literal::List(headers)),
                ("body", Literal::String(self.body)),
            ]),
        })
    }
}

pub struct TurbinaHttpResponse {
    pub status_code: i32,
    pub headers: Vec<String>,
    pub body: String,
}

static HTTP_RESPONSE_STRUCT: LazyLock<StructDefinition> =
    LazyLock::new(|| StructDefinition {
        name: "HttpResponse".to_string(),
        fields: field_map([
            ("statusCode", Type::Int),
            ("headers", Type::String.as_list()),
            ("body", Type::String),
        ]),
    });

impl TurbinaStruct for TurbinaHttpResponse {
    fn get_definition() -> &'static StructDefinition {
        &*&HTTP_RESPONSE_STRUCT
    }

    fn get_type() -> Type {
        Type::StructV2(HTTP_RESPONSE_STRUCT.name.to_string())
    }

    fn to_literal(self) -> Literal {
        let headers: Vec<Literal> = self
            .headers
            .into_iter()
            .map(|h| Literal::String(h))
            .collect();

        Literal::Struct(StructLiteral {
            name: HTTP_RESPONSE_STRUCT.name.to_string(),
            fields: field_map([
                ("statusCode", Literal::Int(self.status_code)),
                ("headers", Literal::List(headers)),
                ("body", Literal::String(self.body)),
            ]),
        })
    }
}
