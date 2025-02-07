// The standard library avaliable anywhere in Turbina

use std::io::{stdout, Write};
use rand::Rng;
use once_cell::sync::Lazy;

use crate::models::{Function, FuncBody, Literal, Scope, Type};


pub static LIBRARY: Lazy<Vec<(&str, Function)>> = Lazy::new(|| {vec![
    ("reverse", Function {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(lib_reverse),
    }),
    ("exit", Function {
        params: vec![("code".into(), Type::Int)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_exit),
    }),
    ("len", Function {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::Int),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Int(text.len().try_into().expect("Integer overflow"))
        }),
    }),
    ("uppercase", Function {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_uppercase())
        }),
    }),
    ("lowercase", Function {
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text)] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_lowercase())
        }),
    }),
    ("printScope", Function {
        params: vec![],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|_, scope| {
            writeln!(stdout(), "{}", scope).unwrap();
            Literal::Null
        }),
    }),
    ("toString", Function {
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            Literal::String(lib_to_string(args))
        }),
    }),
    // TODO: use out stream from program struct instead of using stdout
    ("print", Function {
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, _| {
            write!(stdout(), "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("println", Function {
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, _| {
            writeln!(stdout(), "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("randInt", Function {
        params: vec![("min".into(), Type::Int), ("max".into(), Type::Int)],
        return_type: Some(Type::Int),
        body: FuncBody::Native(|args, _| {
            let [
                Literal::Int(min_arg),
                Literal::Int(max_arg)
            ] = args.as_slice() else {
                panic!("bad args");
            };
            let num = rand::rng().random_range(*min_arg..*max_arg);
            Literal::Int(num)
        }),
    }),
]});

// TODO: create macro for repeated arg unwrapping

fn lib_reverse(args: Vec<Literal>, _: &Scope) -> Literal {
    if let [Literal::String(text)] = args.as_slice() {
        Literal::String(text.chars().rev().collect())
    } else {
        panic!("bad args");
    }
}

fn lib_exit(args: Vec<Literal>, _: &Scope) -> Literal {
    if let [Literal::Int(code)] = args.as_slice() {
        std::process::exit(*code);
    } else {
        panic!("bad args");
    }
}

fn lib_to_string(args: Vec<Literal>) -> String {
    let [data] = args.as_slice() else {
        panic!("bad args");
    };
    match data {
        // to_string() on string literals adds unwanted quotes
        Literal::String(s) => s.clone(),
        _ => data.to_string(),
    }
}

#[cfg(test)]
mod test_library {
    use rstest::rstest;

    use crate::{evaluator::evaluate, models::{Literal, Program}, parser::test_utils::make_tree, validator::validate};

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

    #[rstest]
    #[case("4", "4")]
    #[case(r#""a string""#, "a string")]
    #[case("true", "true")]
    #[case("false", "false")]
    fn test_to_string(#[case] argument: &str, #[case] expected: &str) {
        assert_eq!(
            run_cmd(&format!("toString({});", argument)),
            Literal::String(expected.into())
        )
    }

    #[test]
    fn test_rand_int() {
        // repeat 10 times to account for randomness
        for _ in 0..10 {
            let result = run_cmd("randInt(0, 123);");
            let Literal::Int(rand_int) = result else {
                panic!("expected int result, got {}", result);
            };
            assert!(0 <= rand_int);
            assert!(rand_int < 123);
        }
    }

    fn run_cmd(input: &str) -> Literal{
        let mut program = Program::init();
        let syntax_tree = make_tree(input);
        validate(&program, &syntax_tree).unwrap();
        evaluate(&mut program, &syntax_tree)
    }
}
