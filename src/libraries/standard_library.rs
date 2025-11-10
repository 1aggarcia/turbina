// The standard library available anywhere in Turbina

use std::io::Write;
use std::fs;
use rand::Rng;
use once_cell::sync::Lazy;

use crate::libraries::io::{append_to_file, call_exec, get_filenames_in_directory, open_tcp_server};
use crate::libraries::factories::{create_result_from_error, create_result_from_success, generic_list, generic_type};
use crate::{evaluator::eval_func_call, models::{EvalContext, FuncBody, Function, Literal, Type}};

// TODO: split this up for list, string, I/O, etc.
pub static STANDARD_LIBRARY: Lazy<Vec<(&str, Function)>> = Lazy::new(|| {vec![
    ("reverse", Function {
        type_params: vec![],
        params: define_params![text = Type::String],
        return_type: Some(Type::String),
        body: FuncBody::Native(lib_reverse),
    }),
    ("exit", Function {
        type_params: vec![],
        params: define_params![code = Type::Int],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_exit),
    }),
    ("exec", Function {
        type_params: vec![],
        params: define_params![
            command = Type::String,
            args = Type::String.as_list(),
        ],
        return_type: Some(
            Type::func(
                &[
                    // success handler: stdout and stderr streams
                    Type::func(&EXEC_SUCCESS_TYPES, generic_type("T")),
                    // error handler
                    Type::func(&[Type::String], generic_type("T")),
                ],
                generic_type("T")
            )
        ),
        body: FuncBody::Native(lib_exec),
    }),
    ("help", Function {
        type_params: vec![],
        params: define_params![libraryFunctionName = Type::String],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_help),
    }),
    ("len", Function {
        type_params: vec![],
        params: define_params![text = Type::String],
        return_type: Some(Type::Int),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text), ..] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Int(text.len().try_into().expect("Integer overflow"))
        }),
    }),
    ("uppercase", Function {
        type_params: vec![],
        params: define_params![text = Type::String],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text), ..] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_uppercase())
        }),
    }),
    ("lowercase", Function {
        type_params: vec![],
        params: define_params![text = Type::String], 
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            let [Literal::String(text), ..] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::String(text.to_lowercase())
        }),
    }),
    ("includes", Function {
        type_params: vec![],
        params: define_params![
            text = Type::String,
            substring = Type::String,
        ],
        return_type: Some(Type::Bool),
        body: FuncBody::Native(|args, _| {
            let [
                Literal::String(text),
                Literal::String(substring),
                ..
            ] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Bool(text.contains(substring))
        }),
    }),
    ("startsWith", Function {
        type_params: vec![],
        params: define_params![
            text = Type::String,
            substring = Type::String
        ],
        return_type: Some(Type::Bool),
        body: FuncBody::Native(|args, _| {
            let [
                Literal::String(text),
                Literal::String(substring),
                ..
            ] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Bool(text.starts_with(substring))
        }),
    }),
    ("endsWith", Function {
        type_params: vec![],
        params: define_params![
            text = Type::String,
            substring = Type::String,
        ],
        return_type: Some(Type::Bool),
        body: FuncBody::Native(|args, _| {
            let [
                Literal::String(text),
                Literal::String(substring),
                ..
            ] = args.as_slice() else {
                panic!("bad args");
            };
            Literal::Bool(text.ends_with(substring))
        }),
    }),
    ("split", Function {
        type_params: vec![],
        params: define_params![
            text =Type::String,
            delimiter = Type::String,
        ],
        return_type: Some(Type::String.as_list()),
        body: FuncBody::Native(|args, _| Literal::List(lib_split(args))),
    }),
    ("join", Function {
        type_params: vec![],
        params: define_params![
            list = Type::String.as_list(),
            separator = Type::String,
        ],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| Literal::String(lib_join(args)))
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
        params: define_params![data = Type::Unknown],
        return_type: Some(Type::String),
        body: FuncBody::Native(|args, _| {
            Literal::String(lib_to_string(args))
        }),
    }),
    ("print", Function {
        type_params: vec![],
        params: define_params![data = Type::Unknown],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, context| {
            write!(context.output.stdout, "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("println", Function {
        type_params: vec![],
        params: define_params![data = Type::Unknown],
        return_type: Some(Type::Null),
        body: FuncBody::Native(|args, context| {
            writeln!(context.output.stdout, "{}", lib_to_string(args)).unwrap();
            Literal::Null
        }),
    }),
    ("randInt", Function {
        type_params: vec![],
        params: define_params![min = Type::Int, max = Type::Int],
        return_type: Some(Type::Int),
        body: FuncBody::Native(|args, _| {
            let [
                Literal::Int(min_arg),
                Literal::Int(max_arg),
                ..
            ] = args.as_slice() else {
                panic!("bad args");
            };
            let num = rand::rng().random_range(*min_arg..*max_arg);
            Literal::Int(num)
        }),
    }),
    ("map", Function {
        type_params: vec!["T".into(), "R".into()],
        params: define_params![
            list = generic_list("T"),
            mapFunc = Type::func(
                &[generic_type("T")],
                generic_type("R")
            )
        ],
        return_type: Some(generic_list("R")),
        body: FuncBody::Native(lib_map)
    }),
    ("filter", Function {
        type_params: vec!["T".into()],
        params: define_params![
            list = generic_list("T"),
            predicate = Type::func(
                &[generic_type("T")],
                Type::Bool
            )
        ],
        return_type: Some(generic_list("T")),
        body: FuncBody::Native(lib_filter)
    }),
    ("reduce", Function {
        type_params: vec!["T".into(), "R".into()],
        params: define_params![
            list = generic_list("T"),
            reducer = Type::func(
                &[generic_type("R"), generic_type("T")],
                generic_type("R")
            ),
            initValue = Type::Generic("R".into()),
        ],
        return_type: Some(generic_type("R")),
        body: FuncBody::Native(lib_reduce)
    }),
    ("any", Function {
        type_params: vec!["T".into()],
        params: define_params![
            list = generic_list("T"),
            predicate = Type::func(
                &[generic_type("T")],
                Type::Bool
            ),
        ],
        return_type: Some(Type::Bool),
        body: FuncBody::Native(lib_any)
    }),
    ("every", Function {
        type_params: vec!["T".into()],
        params: define_params![
            list = generic_list("T"),
            predicate = Type::func(
                &[generic_type("T")],
                Type::Bool
            ),
        ],
        return_type: Some(Type::Bool),
        body: FuncBody::Native(lib_every)
    }),
    ("makeList", Function {
        type_params: vec!["T".into()],
        params: define_params![
            length = Type::Int,
            elemFunc = Type::func(
                &[Type::Int],
                generic_type("T")
            )
        ],
        return_type: Some(generic_list("T")),
        body: FuncBody::Native(|args, context| {
            let [
                Literal::Int(length),
                Literal::Closure(elem_func),
                ..
            ] = args.as_slice() else {
                panic!("bad args");
            };
            if *length < 0 {
                panic!("Cannot create a list with a negative length");
            }
            let mut list = Vec::<Literal>::new();
            for i in 0..*length {
                let elem = eval_func_call(
                    context,
                    elem_func.clone(),
                    vec![Literal::Int(i)]
                );
                list.push(elem);
            }
            Literal::List(list)
        })
    }),
    ("appendFile", Function {
        type_params: vec![],
        params: define_params![
            filepath = Type::String,
            contents = Type::String,
        ],
        // return null on success, error message on error
        return_type: Some(Type::String.as_nullable()),
        body: FuncBody::Native(lib_append_file)
    }),
    ("readDir", Function {
        type_params: vec![],
        params: define_params![directoryPath = Type::String],
        return_type: Some(
            Type::func(
                &[
                    // success handler: list of filepaths
                    Type::func(&*READ_DIR_SUCCESS_TYPES, generic_type("T")),
                    // error handler
                    Type::func(&[Type::String], generic_type("T")),
                ],
                generic_type("T")
            )
        ),
        body: FuncBody::Native(lib_read_dir),
    }),
    ("readFile", Function {
        type_params: vec![],
        params: define_params![filepath = Type::String],
        return_type: Some(
            Type::func(
                &[
                    Type::func(&READ_FILE_SUCCESS_TYPES, generic_type("T")),
                    Type::func(&[Type::String], generic_type("T")),
                ],
                generic_type("T")
            )
        ),
        body: FuncBody::Native(lib_read_file)
    }),
    ("writeFile", Function {
        type_params: vec![],
        params: define_params![
            filepath = Type::String,
            contents = Type::String,
        ],
        // return null on success, error message on error
        return_type: Some(Type::String.as_nullable()),
        body: FuncBody::Native(lib_write_file)
    }),
    ("serve", Function {
        type_params: vec![],
        params: define_params![
            address = Type::String,
            handleRequest = Type::func(&[Type::String], Type::String),
            handleError = Type::func(&[Type::String], Type::Null),
        ],
        return_type: Some(Type::Null),
        body: FuncBody::Native(lib_serve)
    })
]});

// TODO: create macro for repeated arg unwrapping

fn lib_reverse(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
    if let [Literal::String(text), ..] = args.as_slice() {
        Literal::String(text.chars().rev().collect())
    } else {
        panic!("bad args");
    }
}

fn lib_exit(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
    if let [Literal::Int(code), ..] = args.as_slice() {
        std::process::exit(*code);
    } else {
        panic!("bad args");
    }
}

static EXEC_SUCCESS_TYPES: [Type; 2] = [Type::String, Type::String];

fn lib_exec(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::String(command),
        Literal::List(args),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    let parsed_args: Vec<&String> = args.into_iter().map(|arg|
        match arg {
            Literal::String(s) => s,
            other => panic!("bad arg: {}", other),
        }
    ).collect();

    match call_exec(&command, &parsed_args) {
        Ok(output) => create_result_from_success(
            vec![
                Literal::String(output.stdout),
                Literal::String(output.stderr)
            ],
            &EXEC_SUCCESS_TYPES,
            context
        ),
        Err(err) => create_result_from_error(
            Literal::String(err.to_string()),
            &EXEC_SUCCESS_TYPES,
            context
        )
    }
}

fn lib_help(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::String(library_function_name),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    match get_function_signature(&library_function_name) {
        Some(signature) => writeln!(context.output.stdout, "{signature}"),
        None => writeln!(
            context.output.stdout,
            "No library function exists with name '{}'",
            library_function_name
        )
    }.expect("stdout stream is broken");

    Literal::Null
}

fn lib_to_string(args: Vec<Literal>) -> String {
    let [
        data,
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    match data {
        // to_string() on string literals adds unwanted quotes
        Literal::String(s) => s.clone(),
        _ => data.to_string(),
    }
}

fn lib_split(args: Vec<Literal>) -> Vec<Literal> {
    let [
        Literal::String(text),
        Literal::String(delimiter),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };

    text
        .split(delimiter)
        .map(|part| Literal::String(part.to_string()))
        .collect::<Vec<Literal>>()
}

fn lib_join(args: Vec<Literal>) -> String {
    let [
        Literal::List(list),
        Literal::String(separator),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };

    list
        .into_iter()
        .map(|literal| match literal {
            Literal::String(s) => s.clone(),
            // the function should only accept string lists, but just in case
            // convert other types to string
            other => other.to_string()
        })
        .collect::<Vec<String>>()
        .join(&separator)
}

fn lib_map(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list_ref),
        Literal::Closure(map_func_ref),
        ..
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
        ..
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
        ..
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

fn lib_any(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list),
        Literal::Closure(predicate),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    let result = list
        .iter()
        .any(|elem|
            eval_func_call(
                context,
                predicate.clone(),
                vec![elem.clone()]
            ) == Literal::Bool(true)
        );
    Literal::Bool(result)
}

fn lib_every(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::List(list),
        Literal::Closure(predicate),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    let result = list
        .iter()
        .all(|elem|
            eval_func_call(
                context,
                predicate.clone(),
                vec![elem.clone()]
            ) == Literal::Bool(true)
        );
    Literal::Bool(result)
}

fn lib_append_file(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
    let [
        Literal::String(filepath),
        Literal::String(contents),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    match append_to_file(filepath, contents) {
        Ok(_) => Literal::Null,
        Err(error) => Literal::String(error.to_string())
    }
}

static READ_FILE_SUCCESS_TYPES: [Type; 1] = [Type::String];

fn lib_read_file(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::String(filepath),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };

    match fs::read_to_string(filepath) {
        Ok(contents) => create_result_from_success(
            vec![Literal::String(contents)],
            &READ_FILE_SUCCESS_TYPES,
            context
        ),
        Err(error) => create_result_from_error(
            Literal::String(error.to_string()),
            &READ_FILE_SUCCESS_TYPES,
            context
        )
    }
}

static READ_DIR_SUCCESS_TYPES: Lazy<[Type; 1]> = Lazy::new(|| [Type::String.as_list()]);

fn lib_read_dir(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::String(path),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };

    let read_dir_result = get_filenames_in_directory(path)
        .map(|filepaths|
            filepaths
                .into_iter()
                .map(|path| Literal::String(path))
                .collect()
        );

    match read_dir_result {
        Ok(filepaths) => create_result_from_success(
            vec![Literal::List(filepaths)],
            &*READ_DIR_SUCCESS_TYPES,
            &context
        ),
        Err(err) => create_result_from_error(
            Literal::String(err.to_string()),
            &*READ_DIR_SUCCESS_TYPES,
            &context
        ),
    }
}

fn lib_write_file(args: Vec<Literal>, _: &mut EvalContext) -> Literal {
    let [
        Literal::String(filepath),
        Literal::String(contents),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };
    match fs::write(filepath, contents) {
        Ok(_) => Literal::Null,
        Err(error) => Literal::String(error.to_string())
    }
}

fn lib_serve(args: Vec<Literal>, context: &mut EvalContext) -> Literal {
    let [
        Literal::String(address),
        Literal::Closure(handle_request),
        Literal::Closure(handle_error),
        ..
    ] = args.as_slice() else {
        panic!("bad args");
    };

    let handle_tcp_request = |request: String| {
        let response = eval_func_call(
            context,
            handle_request.clone(),
            vec![Literal::String(request)]
        );
        match response {
            Literal::String(s) => s,
            other => panic!("bad return type of TCP handler: {}", other),
        } 
    };

    match open_tcp_server(address, handle_tcp_request) {
        Ok(_) => {}
        Err(err) => {
            eval_func_call(
                context,
                handle_error.clone(),
                vec![Literal::String(err.to_string())]
            );
        }
    };
    Literal::Null
}

fn get_function_signature(function_name: &str) -> Option<String> {
    let (name, func) = STANDARD_LIBRARY
        .iter()
        .find(|(n, _)| *n == function_name)?;

    let type_param_list = if func.type_params.is_empty() {
        String::new()
    } else {
        format!("<{}>", func.type_params.join(", "))
    };

    let param_list = func.params
        .iter()
        .map(|(param, datatype)| format!("{param}: {datatype}"))
        .collect::<Vec<String>>()
        .join(", ");

    let return_type = match &func.return_type {
        Some(datatype) => datatype.to_string(),
        None => "unknown".into(),
    };

    format!("{name}{type_param_list}({param_list}): {return_type}").into()
}

#[cfg(test)]
mod test_library {
    use super::*;

    use rstest::rstest;

    use crate::{evaluator::evaluate, models::{Literal, Program}, parser::test_utils::make_tree, type_resolver::resolve_type};

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
    #[case("racecar", "race", true)]
    #[case("racecar", "car", true)]
    #[case("racecar", "ceca", true)]
    #[case("racecar", "build", false)]
    fn test_includes(
        #[case] input: &str,
        #[case] substring: &str,
        #[case] expected: bool
    ) {
        let formatted_input =
            format!(r#"includes("{}", "{}");"#, input, substring); 
        assert_eq!(
            run_cmd(&formatted_input),
            Literal::Bool(expected)
        ); 
    }
    

    #[rstest]
    #[case("racecar", "race", true)]
    #[case("racecar", "car", false)]
    #[case("RACECAR", "race", false)]
    fn test_starts_with(
        #[case] input: &str,
        #[case] substring: &str,
        #[case] expected: bool
    ) {
        let formatted_input =
            format!(r#"startsWith("{}", "{}");"#, input, substring); 
        assert_eq!(
            run_cmd(&formatted_input),
            Literal::Bool(expected)
        ); 
    }

    #[rstest]
    #[case("racecar", "race", false)]
    #[case("racecar", "car", true)]
    #[case("RACECAR", "car", false)]
    fn test_ends_with(
        #[case] input: &str,
        #[case] substring: &str,
        #[case] expected: bool
    ) {
        let formatted_input =
            format!(r#"endsWith("{}", "{}");"#, input, substring); 
        assert_eq!(
            run_cmd(&formatted_input),
            Literal::Bool(expected)
        ); 
    }

    #[test]
    fn test_split() {
        let expected: Vec<Literal> = vec![1, 2, 3, 4]
            .iter()
            .map(|num| Literal::String(num.to_string()))
            .collect();

        assert_eq!(
            run_cmd(r#"split("1;2;3;4", ";");"#),
            Literal::List(expected)
        );
    }

    #[rstest]
    #[case(r#"["a", "b", "c"]"#, r#""-""#, "a-b-c")]
    #[case(r#"["1", "2", "3"]"#, r#""""#, "123")]
    #[case(r#"["x", "y", "z"]"#, r#"", ""#, "x, y, z")]
    fn test_join(
        #[case] input: &str,
        #[case] separator: &str,
        #[case] expected: &str
    ) {
        assert_eq!(
            run_cmd(&format!("join({input}, {separator});")),
            Literal::String(expected.into())
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
    #[case::any_returns_true_if_single_match(
        "any([1, 12, 3], (x: int) -> x > 10);",
        true,
    )]
    #[case::every_returns_false_if_single_match(
        "every([1, 12, 3], (x: int) -> x > 10);",
        false,
    )]
    #[case::any_returns_false_if_no_matches(
        "any([-1, -2, -3], (x: int) -> x > 0);",
        false
    )]
    #[case::every_returns_true_if_all_match(
        "every([-1, -2, -3], (x: int) -> x < 0);",
        true 
    )]
    #[case::any_returns_false_if_empty_list(
        "any([], (x: string) -> len(x) == 15);",
        false 
    )]
    #[case::every_returns_true_if_empty_list(
        "every([], (x: string) -> len(x) == 15);",
        true
    )]
    fn test_list_predicates(
        #[case] input: &str,
        #[case] expected: bool
    ) {
        assert_eq!(run_cmd(input), Literal::Bool(expected))
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

    #[test]
    fn test_make_int_list() {
        let expected = [0, 1, 2, 3, 4]
            .iter().map(|i| Literal::Int(*i)).collect();
        assert_eq!(
            run_cmd("makeList(5, (i: int) -> i);"),
            Literal::List(expected)
        )
    }

    #[test]
    fn test_make_string_list() {
        let expected = ["", "", "", "", "", "", "", "", ""]
            .iter().map(|i| Literal::String(i.to_string())).collect();
        assert_eq!(
            run_cmd(r#"makeList(9, () -> "");"#),
            Literal::List(expected)
        )
    }

    #[test]
    fn test_get_function_signature() {        
        let expected = "map<T, R>(list: T[], mapFunc: T -> R): R[]";

        assert_eq!(
            get_function_signature("map"),
            Some(expected.to_string())
        );
    }

    fn run_cmd(input: &str) -> Literal {
        let mut program = Program::init_with_std_streams();
        let syntax_tree = make_tree(input);
        resolve_type(&program, &syntax_tree).unwrap();
        evaluate(&mut program, &syntax_tree).unwrap()
    }
}
