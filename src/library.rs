// The standard library avaliable anywhere in Turbina

use crate::models::{Func, FuncBody, Literal, Scope, Type};

use once_cell::sync::Lazy;

pub static LIBRARY: Lazy<Vec<(&str, Func)>> = Lazy::new(|| {vec![
    ("reverse", Func {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(lib_reverse),
    }),
    ("exit", Func {
        params: vec![("code".into(), Type::Int)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_exit),
    }),
    ("len", Func {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::Int),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Int(text.len().try_into().expect("Integer overflow"))
        }),
    }),
    ("uppercase", Func {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_uppercase())
        }),
    }),
    ("lowercase", Func {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_lowercase())
        }),
    }),
    ("printScope", Func {
        params: vec![],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|_, scope| {
            println!("{}", scope);
            Literal::Null
        })
    })
]});

// TODO: create macro for repeated arg unwrapping

fn lib_reverse(args: Vec<Literal>, _: &Scope) -> Literal {
    if let [Literal::String(text)] = args.as_slice() {
        Literal::String(text.chars().rev().collect())
    } else {
        panic!("bad args");
    }
}

fn lib_exit(args: Vec<Literal>, _: &Scope) -> Literal{
    if let [Literal::Int(code)] = args.as_slice() {
        std::process::exit(*code);
    } else {
        panic!("bad args");
    }
}

#[cfg(test)]
mod test_library {
    use crate::{evaluator::evaluate, lexer::tokenize, models::{Literal, Program}, parser::parse, validator::validate};

    #[test]
    fn test_reverse() {
        assert_eq!(
            run_cmd(r#"reverse("abcde");"#),
            Literal::String("edcba".into())
        );
    }

    #[test]
    fn test_len() {
        assert_eq!(run_cmd(r#"len("");"#), Literal::Int(0));
        assert_eq!(run_cmd(r#"len("1234567");"#), Literal::Int(7));
    }

    #[test]
    fn test_uppercase() {
        assert_eq!(
            run_cmd(r#"uppercase("Turbina");"#),
            Literal::String("TURBINA".into())
        );
    }

    #[test]
    fn test_lowercase() {
        assert_eq!(
            run_cmd(r#"lowercase("John DOE");"#),
            Literal::String("john doe".into())
        );
    }

    fn run_cmd(input: &str) -> Literal{
        let mut program = Program::init();
        let tokens = tokenize(input).unwrap();
        let syntax_tree = parse(tokens).unwrap();
        validate(&program, &syntax_tree).unwrap();
        evaluate(&mut program, &syntax_tree)
    }
}
