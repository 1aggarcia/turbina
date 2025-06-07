// The standard library available anywhere in Turbina

use std::io::Write;
use rand::Rng;
use once_cell::sync::Lazy;

use crate::{evaluator::eval_func_call, models::{EvalContext, FuncBody, Function, Literal, Type}};

pub static LIBRARY: Lazy<Vec<(&str, Function)>> = Lazy::new(|| {vec![
    ("reverse", Function {
        type_params: vec![],
        params: vec![("text".into(), Type::String)],
        return_type: Some(Type::String),
        body: FuncBody::Native(lib_reverse),
    }),
    ("exit", Function {
        type_params: vec![],
        params: vec![("code".into(), Type::Int)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_exit),
    }),
    ("len", Function {
        type_params: vec![],
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
        type_params: vec![],
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
        type_params: vec![],
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
        type_params: vec![],
        params: vec![],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|_, context| {
            writeln!(context.output.stdout, "{}", context.scope).unwrap();
            Literal::Null
        }),
    }),
    ("toString", Function {
        type_params: vec![],
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            Literal::String(lib_to_string(args))
        }),
    }),
    ("print", Function {
        type_params: vec![],
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, context| {
            write!(context.output.stdout, "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("println", Function {
        type_params: vec![],
        params: vec![("data".into(), Type::Unknown)],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, context| {
            writeln!(context.output.stdout, "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("randInt", Function {
        type_params: vec![],
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
    ("map", Function {
        type_params: vec!["T".into(), "R".into()],
        params: vec![
            ("list".into(), generic_list("T")),
            ("mapFunc".into(), Type::func(
                &[generic_type("T")],
                generic_type("R")
            ))
        ],
        return_type: Some(generic_list("R")),
        body: FuncBody::Native(lib_map)
    }),
    ("filter", Function {
        type_params: vec!["T".into()],
        params: vec![
            ("list".into(), generic_list("T")),
            ("predicate".into(), Type::func(
                &[generic_type("T")],
                Type::Bool
            ))
        ],
        return_type: Some(generic_list("T")),
        body: FuncBody::Native(lib_filter)
    }),
    // type signature: (T[], (R, T) -> R, R) -> R
    ("reduce", Function {
        type_params: vec!["T".into(), "R".into()],
        params: vec![
            ("list".into(), generic_list("T")),
            ("reducer".into(), Type::func(
                &[generic_type("R"), generic_type("T")],
                Type::Generic("R".into())
            )),
            ("initValue".into(), Type::Generic("R".into())),
        ],
        return_type: Some(generic_type("R")),
        body: FuncBody::Native(lib_reduce)
    }),
]});

// TODO: create macro for repeated arg unwrapping

fn lib_reverse(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
    if let [Literal::String(text)] = args.as_slice() {
        Literal::String(text.chars().rev().collect())
    } else {
        panic!("bad args");
    }
}

fn lib_exit(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
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

fn lib_map(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list_ref),
        Literal::Closure(map_func_ref), 
    ] = args.as_slice() else {
        panic!("bad args");
    };
    let list = list_ref.clone();
    let map_func = map_func_ref.clone();
    let transformed = list
        .iter()
        .map(|elem| eval_func_call(
            context,
            map_func.clone(),
            vec![elem.to_owned()]
        ))
        .collect();
    Literal::List(transformed)
}

fn lib_filter(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list_ref),
        Literal::Closure(predicate_ref), 
    ] = args.as_slice() else {
        panic!("bad args");
    };
    let list = list_ref.clone();
    let predicate = predicate_ref.clone();
    let mut transformed = Vec::with_capacity(list.len());
    for elem in list {
        let should_include =
            eval_func_call(context, predicate.clone(), vec![elem.clone()]);
        if should_include == Literal::Bool(true) {
            transformed.push(elem.clone());
        }
    }
    Literal::List(transformed)
}

fn lib_reduce(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list),
        Literal::Closure(reducer),
        init_value,
    ] = args.as_slice() else {
        panic!("bad args");
    };
    list
        .iter()
        .fold(init_value.clone(), |accumulator, elem| eval_func_call(
            context,
            reducer.clone(),
            vec![accumulator.to_owned(), elem.clone()]
        ))
}

fn generic_type(type_name: &str) -> Type {
    Type::Generic(type_name.to_string())
}

fn generic_list(type_name: &str) -> Type {
    Type::Generic(type_name.to_string()).to_list()
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

    #[rstest]
    #[case::map1(
        "map([65, 44, 12], (x: int) -> x * 10);",
        &[650, 440, 120]
    )]
    #[case::map2(
        "map([1, 2, 3, 4, 5], (x: int) -> x % 2);",
        &[1, 0, 1, 0, 1]
    )]
    #[case::map3(
        r#"map(["a", "bcd", "ef"], (s: string) -> len(s));"#,
        &[1, 3, 2]
    )]
    #[case::filter(
        "filter([1, 65, 54, 12], (x: int) -> x < 50);",
        &[1, 12]
    )]
    fn test_list_transformations(
        #[case] input: &str,
        #[case] expected: &[i32],
    ) {
        let expected_list =
            Literal::List(expected.iter().map(|i| Literal::Int(*i)).collect());
        assert_eq!(run_cmd(input), expected_list);
    }

    #[rstest]
    #[case::empty_list(
        "reduce([], (x: int, y: int) -> 0, 15);",
        15
    )]
    #[case::nonempty_list(
        "reduce([1, 2, 3, 4], (x: int, y: int) -> x + y, 10);",
        20
    )]
    #[case::max(
        "reduce([5, 8, 1, 6], (x: int, y: int) -> if (x > y) x else y, 0);",
        8
    )]
    fn test_reduce(#[case] input: &str, #[case] expected: i32) {
        assert_eq!(run_cmd(input), Literal::Int(expected));
    }

    fn run_cmd(input: &str) -> Literal {
        let mut program = Program::init_with_std_streams();
        let syntax_tree = make_tree(input);
        validate(&program, &syntax_tree).unwrap();
        evaluate(&mut program, &syntax_tree).unwrap()
    }
}
