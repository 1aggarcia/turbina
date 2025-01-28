// The standard library avaliable anywhere in Turbina

use crate::models::{Func, FuncBody, Literal, Type};

use once_cell::sync::Lazy;

pub static LIBRARY: Lazy<Vec<(&str, Func)>> = Lazy::new(|| {vec![
    ("reverse", Func {
        params: vec![("text".into(), Type::String)],
        return_type: Type::String,
        body: FuncBody::Native(lib_reverse),
    }),
    ("exit", Func {
        params: vec![("code".into(), Type::Int)],
        return_type: Type::Null,
        body: FuncBody::Native(lib_exit),
    })
]});

// TODO: create macro for repeated arg unwrapping

fn lib_reverse(args: Vec<Literal>) -> Literal {
    if let [Literal::String(text)] = args.as_slice() {
        Literal::String(text.chars().rev().collect())
    } else {
        panic!("bad args");
    }
}

fn lib_exit(args: Vec<Literal>) -> Literal{
    if let [Literal::Int(code)] = args.as_slice() {
        std::process::exit(*code);
    } else {
        panic!("bad args");
    }
}

#[cfg(test)]
mod test_library {
    use crate::models::Literal;

    use super::lib_reverse;

    #[test]
    fn test_lib_reverse() {
        assert_eq!(
            lib_reverse(vec![Literal::String("rust".into())]),
            Literal::String("tsur".into())
        );
    }
}
