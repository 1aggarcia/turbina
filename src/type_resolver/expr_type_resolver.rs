use std::collections::HashMap;

use crate::errors::{error, InterpreterError, MultiResult};
use crate::models::{BinaryExpr, BinaryOp, CodeBlock, CondExpr, Expr, FuncBody, FuncCall, Function, Literal, Term, Type};
use crate::type_resolver::shared::{SubResult, TypeContext};
use crate::type_resolver::resolve_statement_type;

pub fn resolve_expr_type(context: &TypeContext, expr: &Expr) -> SubResult {
    match expr {
        Expr::Binary(b) => resolve_binary_expr_type(context, b),
        Expr::CodeBlock(b) => resolve_code_block_type(context, b),
        Expr::Cond(c) => resolve_cond_expr_type(context, c),
        Expr::Function(f) => resolve_function_type(context, f),
    }
}

/// Check that the types for every term in the expression are valid
fn resolve_binary_expr_type(context: &TypeContext, expr: &BinaryExpr) -> SubResult {
    let mut errors = Vec::<InterpreterError>::new();
    let mut result = None;

    match resolve_term_type(context, &expr.first) {
        Ok(t) => result = Some(t),
        Err(e) => errors.extend(e),
    }

    for (op, term) in &expr.rest {
        let result_type = match result {
            Some(ref t) => t.clone(),
            None => continue,
        };
        // TODO: maybe use mutable type contexts instead
        let operator_context = if *op == BinaryOp::Pipe {
            // create a local variable under "_" to pipe the left side into
            // the right side
            &TypeContext {
                variable_types: &HashMap::from([
                    ("_".into(), result_type.clone())
                ]),
                parameter_types: &HashMap::new(),
                generic_type_parameters: &[],
                name_to_bind: None,
                parent: Some(context)
            }
        } else {
            context
        };
        let new_type = match resolve_term_type(operator_context, &term) {
            Ok(t) => t,
            Err(e) => {
                errors.extend(e);
                continue;
            }
        };

        match binary_op_return_type(result_type, *op, new_type) {
            Ok(t) => result = Some(t),
            Err(e) => errors.extend(e),
        }
    }

    if errors.is_empty() {
        return Ok(result.unwrap());
    } else {
        return Err(errors);
    }
}

fn resolve_code_block_type(context: &TypeContext, block: &CodeBlock) -> SubResult {
    let mut variable_types = HashMap::new();
    let parameter_types = HashMap::new();

    let default_type = Ok(Type::Null);
    block.statements.iter().fold(default_type, |last_result, statement| {
        // propagate the error to the end without checking any more statements
        if matches!(last_result, Err(_)) {
            return last_result;
        }
        let mut statement_context = TypeContext {
            variable_types: &variable_types,
            parameter_types: &parameter_types,
            generic_type_parameters: &[],
            name_to_bind: None,
            parent: Some(context),
        };
        let tree_type = resolve_statement_type(&mut statement_context, &statement)?;

        if let Some(name) = &tree_type.name_to_bind {
            variable_types.insert(name.into(), tree_type.datatype.clone());
        }
        Ok(tree_type.datatype)
    })
}

/// Check that the condition is a boolean type, and the "if" and "else" branches
/// are of the same type
fn resolve_cond_expr_type(context: &TypeContext, expr: &CondExpr) -> SubResult {
    let cond_type = resolve_expr_type(context, &expr.cond)?;
    if cond_type != Type::Bool {
        return Err(InterpreterError::InvalidType {
            datatype: cond_type
        }.into());
    }
    let true_type = resolve_expr_type(context, &expr.if_true)?;
    let false_type = resolve_expr_type(context, &expr.if_false)?;
    let union_type = find_union_type(true_type, false_type);
    Ok(union_type)
}

/// Check two assertions:
/// - That the return type can be resolved with global bindings and new
///     bindings introduced by the input parameters
/// - That all generic types used in the function body are declared in either
///     the function definition or the parent scope.
fn resolve_function_type(context: &TypeContext, function: &Function) -> SubResult {
    // if the type is generic, has it been declared?
    let validate_type_reference = |datatype: &Type| {
        let Type::Generic(type_param) = datatype else {
            return Ok(());
        };
        if function.type_params.contains(&type_param)
                || context.contains_type_parameter(&type_param) {
            Ok(())
        } else {
            Err(InterpreterError::UndeclaredGeneric {
                generic: type_param.into()
            })
        }
    };

    let mut param_types = HashMap::<String, Type>::new();

    // must also be stored as a list since maps don't have order
    let mut param_type_list = Vec::<Type>::new();
    let mut param_errors = Vec::<InterpreterError>::new();

    for (id, datatype) in function.params.clone() {
        if context.contains_parameter(&id) {
            param_errors.push(InterpreterError::ReassignError { id });
            continue;
        }
        if let Err(err) = validate_type_reference(&datatype) {
            param_errors.push(err);
            continue;
        }
        param_types.insert(id.clone(), datatype.clone());
        param_type_list.push(datatype);
    }

    if !param_errors.is_empty() {
        return Err(param_errors);
    }

    let mut func_context = TypeContext {
        variable_types: &mut HashMap::new(),
        parameter_types: &param_types,
        generic_type_parameters: &function.type_params,
        name_to_bind: None,
        parent: Some(&context),
    };

    if let Some(declared_return_type) = &function.return_type {
        if let Err(err) = validate_type_reference(declared_return_type) {
            return Err(err.into());
        }
        let func_type = Type::Func {
            input: param_type_list,
            output: Box::new(declared_return_type.clone())
        };
        let FuncBody::Expr(func_body) = &function.body else {
            return Ok(func_type);
        };

        // add recursive function to type context so it can be used in the body
        let mut variable_types = HashMap::new();
        if let Some(recursive_func_name) = &context.name_to_bind {
            variable_types.insert(
                recursive_func_name.clone(),
                func_type.clone()
            );
            func_context.variable_types = &variable_types;
        }

        let body_type = resolve_expr_type(&func_context, &func_body)?;
        if !body_type.is_assignable_to(&declared_return_type) {
            return Err(InterpreterError
                ::bad_return_type(declared_return_type, &body_type).into());
        }
        Ok(func_type)
    } else {
        let FuncBody::Expr(func_body) = &function.body else {
            // this should never happen
            let message = "Function type is undecidable".to_string();
            return Err(InterpreterError::TypeError { message }.into());
        };

        let func_type = Type::Func {
            input: param_type_list,
            output: Box::new(resolve_expr_type(&func_context, &func_body)?)
        };
        Ok(func_type)
    }
}

/// For literals, check that the type matches any unary operators.
/// 
/// For non-literals, check that they exist and their type matches any
/// unary operators applied (! and -).
fn resolve_term_type(context: &TypeContext, term: &Term) -> SubResult {
    match term {
        Term::Literal(lit) => get_literal_type(lit),
        Term::Id(id) => resolve_id_type(context, &id),
        Term::Not(term) => resolve_negated_bool_type(context, term),
        Term::Minus(term) => validated_negated_int(context, term),
        Term::Expr(expr) => resolve_expr_type(context, expr),
        Term::FuncCall(call) => resolve_func_call_type(context, call),
        Term::List(list) => resolve_list_type(context, list),
        Term::NotNull(term) => {
            let datatype = resolve_term_type(context, term)?;
            if let Type::Nullable(inner_type) = datatype {
                return Ok(*inner_type);
            }
            Ok(datatype)
        },
    }
}

/// Should not be called with closures, closures are only created at runtime.
fn get_literal_type(literal: &Literal) -> SubResult {
    let datatype = match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
        Literal::Null => Type::Null,
        Literal::List(list) =>
            panic!("Literal list created before evaluation: {:?}", list), 
        Literal::Closure(closure) =>
            panic!("Closure created before evaluation: {:?}", closure),
    };
    Ok(datatype)
}

/// Check that the id exists in the program's type environment
/// The id may not have an associated value until evaluation
fn resolve_id_type(context: &TypeContext, id: &String) -> SubResult {
    context.lookup(id).ok_or(error::undefined_id(id).into())
}

/// Check that the passed in term is a boolean
fn resolve_negated_bool_type(context: &TypeContext, inner_term: &Term) -> SubResult {
    let datatype = resolve_term_type(context, inner_term)?;
    match datatype {
        Type::Bool => Ok(datatype),
        _ => Err(vec![error::unary_op_type("!", datatype)])
    }
}

/// Check that the passed in term is an int
fn validated_negated_int(context: &TypeContext, inner_term: &Term) -> SubResult {
    let datatype = resolve_term_type(context, inner_term)?;
    match datatype {
        Type::Int=> Ok(datatype),
        _ => Err(vec![error::unary_op_type("-", datatype)])
    }
}



/// Check that the function being called is defined and the input types match
/// the argument list
fn resolve_func_call_type(context: &TypeContext, call: &FuncCall) -> SubResult {
    let (param_types, output_type) = match resolve_term_type(context, &call.func)? {
        Type::Func { input, output } => (input, output),
        _ => {
            let err = InterpreterError::not_a_function(&call.func); 
            return Err(vec![err]);
        }
    };
    if call.args.len() != param_types.len() {
        let err = InterpreterError::ArgCount {
            got: call.args.len(),
            expected: param_types.len()
        };
        return Err(vec![err]);
    }

    let mut errors = vec![];
    for (arg, param_type) in call.args.iter().zip(param_types.iter()) {
        let arg_type = resolve_expr_type(context, arg)?;
        if arg_type.is_assignable_to(param_type) {
            continue;
        }
        errors.push(InterpreterError::UnexpectedType {
            got: arg_type,
            expected: param_type.clone()
        });
    }

    if errors.is_empty() { Ok(*output_type) } else { Err(errors) }
}

/// Determines the strictest type that applies to all elements in the list
fn resolve_list_type(context: &TypeContext, list: &Vec<Expr>) -> SubResult {
    let list_type = list
        .iter()
        .map(|element| resolve_expr_type(context, element))  
        .collect::<MultiResult<Vec<Type>>>()?
        .into_iter()
        .reduce(find_union_type)
        .map(|union_type| union_type.as_list())
        .unwrap_or(Type::EmptyList);

    Ok(list_type)
}

/// Returns the strictest type possible that includes both input types
fn find_union_type(type1: Type, type2: Type) -> Type {
    if type1 == type2 {
        type1
    } else if type1.is_assignable_to(&type2) {
        type2
    } else if type2.is_assignable_to(&type1) {
        type1
    } else if type1 == Type::Null {
        type2.as_nullable()
    } else if type2 == Type::Null {
        type1.as_nullable()
    } else {
        match (type1, type2) {
            (Type::List(inner_type1), Type::List(inner_type2)) =>
                find_union_type(*inner_type1, *inner_type2).as_list(),

            (Type::Nullable(inner_type_1), outer_type2) =>
                find_union_type(*inner_type_1, outer_type2).as_nullable(),

            (outer_type1, Type::Nullable(inner_type2)) =>
                find_union_type(outer_type1, *inner_type2).as_nullable(),

            _ => Type::Unknown
        }
    }
}

/// Get the return type of a binary operator if the left and right operand
/// types are valid, otherwise return a validation error
fn binary_op_return_type(left: Type, operator: BinaryOp, right: Type) -> SubResult {
    use BinaryOp::*;

    let type_error =
        Err(error::binary_op_types(operator, &left, &right).into());

    match operator {
        And | Or => if left == Type::Bool && left == right
            { Ok(Type::Bool) } else { type_error }

        // equality operators
        NotEq | Equals =>
            if left.is_assignable_to(&right) || right.is_assignable_to(&left)
                { Ok(Type::Bool) } else { type_error },

        // number comparison
        GreaterThan | GreaterThanOrEqual | LessThan | LessThanOrEqual =>
            if left == right && left == Type::Int
                { Ok(Type::Bool) } else { type_error },

        // math operators
        Plus => match (&left, right) {
            (Type::String, Type::String)
            | (Type::Int, Type::Int) => Ok(left),
            _ => type_error
        },
        Minus | Percent | Slash | Star =>
            if left == Type::Int { Ok(Type::Int) } else { type_error },

        Pipe => Ok(right),
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::models::test_utils::term_tree;
    use crate::parser::test_utils::make_tree;
    use crate::errors::{error, InterpreterError};
    use crate::models::{BinaryOp, Literal, Program, Term, Type};
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::*;

    mod term {
        use super::*;

        #[rstest]
        #[case(Literal::Int(3), Type::Int)]
        #[case(Literal::String("asdf".to_string()), Type::String)]
        #[case(Literal::Bool(false), Type::Bool)]
        fn returns_ok_for_literals(#[case] literal: Literal, #[case] expected: Type) {
            let tree = term_tree(Term::Literal(literal.clone()));
            assert_eq!(resolve_type_fresh(tree), ok_without_binding(expected));
        }

        #[test]
        fn it_returns_ok_for_valid_symbol() {
            let tree = make_tree("x;");
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".to_string(), Type::Int);

            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::Int));
        }

        #[rstest]
        #[case(Type::String, Type::String)]
        #[case(Type::Int.as_nullable(), Type::Int)]
        fn it_performs_non_null_assertion(
            #[case] symbol_type: Type,
            #[case] casted_type: Type
        ) {
            let tree = make_tree("x!;");
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".to_string(), symbol_type);

            assert_eq!(resolve_type(&program, &tree), ok_without_binding(casted_type));
        }

        #[test]
        fn it_returns_error_for_non_existent_symbol() {
            let tree = make_tree("x;");
            let expected = vec![error::undefined_id("x")];
            assert_eq!(resolve_type_fresh(tree), Err(expected));
        }

        #[rstest]
        #[case("!3;", error::unary_op_type("!", Type::Int))]
        #[case("!\"str\";", error::unary_op_type("!", Type::String))]
        #[case("-false;", error::unary_op_type("-", Type::Bool))]
        #[case("-\"str\";", error::unary_op_type("-", Type::String))]
        fn it_returns_error_for_bad_negated_types(
            #[case] input: &str,
            #[case] error: InterpreterError,
        ) {
            let tree = make_tree(input);
            assert_eq!(resolve_type_fresh(tree), Err(vec![error]));
        }

        #[rstest]
        #[case::string_list(r#"["one", "two", "three"];"#, Type::String)]
        #[case::bool_list("[false, true];", Type::Bool)]
        #[case::many_types(r#"["abc", 123, true];"#, Type::Unknown)]
        #[case::many_types_and_null(
            r#"["abc", null, true];"#,
            Type::Unknown.as_nullable()
        )]
        #[case::null(r#"[null, null, null];"#, Type::Null)]
        #[case::nullable_int_list(
            r#"[1, 4, null, 5];"#,
            Type::Int.as_nullable()
        )]
        fn it_determines_strictest_type_for_lists(
            #[case] input: &str,
            #[case] list_type: Type,
        ) {
            let tree = make_tree(input);
            let expected = Type::List(Box::new(list_type));
            assert_eq!(resolve_type_fresh(tree), ok_without_binding(expected));
        }

        #[test]
        fn it_returns_correct_empty_list_type() {
            let tree = make_tree("[];");
            let expected = Type::EmptyList;
            assert_eq!(resolve_type_fresh(tree), ok_without_binding(expected));
        }
    }

    mod binary_expr {
        use super::*;

        #[rstest]
        #[case("3 + \"\";", BinaryOp::Plus, Type::Int, Type::String)]
        #[case("null + 5;", BinaryOp::Plus, Type::Null, Type::Int)]
        #[case("\"\" - \"\";", BinaryOp::Minus, Type::String, Type::String)]
        #[case("true % false;", BinaryOp::Percent, Type::Bool, Type::Bool)]
        #[case("0 == false;", BinaryOp::Equals, Type::Int, Type::Bool)]
        #[case("\"\" != 1;", BinaryOp::NotEq, Type::String, Type::Int)]

        #[case("2 >  false;", BinaryOp::GreaterThan, Type::Int, Type::Bool)]
        #[case("2 >= \"\";", BinaryOp::GreaterThanOrEqual, Type::Int, Type::String)]
        #[case("2 <  \"\";", BinaryOp::LessThan, Type::Int, Type::String)]
        #[case("2 <= false;", BinaryOp::LessThanOrEqual, Type::Int, Type::Bool)]
        fn it_returns_error_for_illegal_types(
            #[case] input: &str,
            #[case] op: BinaryOp,
            #[case] left_type: Type,
            #[case] right_type: Type
        ) {
            let tree = make_tree(input);
            let expected = error::binary_op_types(op, &left_type, &right_type);
            assert_eq!(resolve_type_fresh(tree), Err(vec![expected]));
        }

        #[rstest]
        // right arg undefined
        #[case("a + 3;", vec![error::undefined_id("a")])]

        // left arg undefined
        #[case("1 + c;", vec![error::undefined_id("c")])]

        // many args undefined - stop after the first one
        #[case("x + y - z;", vec![error::undefined_id("x")])]
        fn it_returns_error_for_child_error(
            #[case] input: &str, #[case] errors: Vec<InterpreterError>
        ) {
            // symbol does not exist
            let tree = make_tree(input);
            assert_eq!(resolve_type_fresh(tree), Err(errors));
        }

        #[rstest]
        #[case("2 + 2;", Type::Int)]
        #[case("2 / 2;", Type::Int)]
        #[case("2 |> toString(_);", Type::String)]
        #[case(r#""a" + "b";"#, Type::String)]
        fn it_returns_ok_for_good_operands(
            #[case] input: &str,
            #[case] evaluated_type: Type
        ) {
            let tree = make_tree(input);
            assert_eq!(resolve_type_fresh(tree), ok_without_binding(evaluated_type))
        }

        #[rstest]
        // equality
        #[case("0 == 1;")]
        #[case("true != false;")]
        #[case("\"a\" == \"b\";")]

        // comparison
        #[case("2 > 2;")]
        #[case("2 >= 2;")]
        #[case("2 < 2;")]
        #[case("2 <= 2;")]

        // compound
        #[case("false && true;")]
        #[case("false || true;")]
        #[case("1 % 3 == 0 && 1 % 5 == 0;")]
        fn it_returns_ok_for_boolean_operator_on_same_type(#[case] input: &str) {
            let tree = make_tree(input);
            let expected = ok_without_binding(Type::Bool);
            assert_eq!(resolve_type_fresh(tree), expected);
        }

        #[rstest]
        #[case(Type::Unknown, Type::Int)]
        #[case(Type::String, Type::String.as_nullable())]
        #[case(Type::Null, Type::String.as_nullable())]
        fn it_returns_ok_for_equality_of_comparable_types(
            #[case] type_a: Type, #[case] type_b: Type
        ) {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".into(), type_a);
            program.type_context.insert("b".into(), type_b);

            let expected = ok_without_binding(Type::Bool);

            // should be transitive
            assert_eq!(resolve_type(&program, &make_tree("a == b;")), expected);
            assert_eq!(resolve_type(&program, &make_tree("a != b;")), expected);
            assert_eq!(resolve_type(&program, &make_tree("b == a;")), expected);
            assert_eq!(resolve_type(&program, &make_tree("b != a;")), expected);
        }

        #[rstest]
        #[case(Type::Int, Type::String)]
        #[case(Type::Int.as_nullable(), Type::String.as_nullable())]
        fn it_returns_error_for_equality_of_disjoint_types(
            #[case] type_a: Type, #[case] type_b: Type
        ) {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".into(), type_a.clone());
            program.type_context.insert("b".into(), type_b.clone());

            let expected_eq = Err(
                vec![error::binary_op_types(BinaryOp::Equals, &type_a, &type_b)]
            );
            let expected_not_eq = Err(
                vec![error::binary_op_types(BinaryOp::NotEq, &type_a, &type_b)]
            );
            assert_eq!(resolve_type(&program, &make_tree("a == b;")), expected_eq);
            assert_eq!(resolve_type(&program, &make_tree("a != b;")), expected_not_eq);
        }
    }

    mod code_block {
        use super::*; 

        #[test]
        fn it_returns_type_of_final_expression() {
            let input = make_tree(r#"{
                let two: int = 2;
                let one: string = "one";
                two + 3
            };"#);
            assert_eq!(resolve_type_fresh(input), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_type_of_final_statement() {
            let input = make_tree(r#"{
                let two: int = 2;
                let one: string = "one";
                two + 3;
            };"#);
            assert_eq!(resolve_type_fresh(input), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_finds_errors_in_local_bindings() {
            let input = make_tree(r#"{
                let an_int = 0;
                let a_string = "";
                an_int + a_string
            };"#);
            let expected = error::binary_op_types(
                BinaryOp::Plus, &Type::Int, &Type::String);
            assert_eq!(resolve_type_fresh(input), Err(expected.into()));
        }

        #[test]
        fn it_stops_validating_statements_after_first_error() {
            let input = make_tree(r#"{
                let bad = 1 + "";
                let good = 1 + 2;
            }"#);
            let expected = error::binary_op_types(
                BinaryOp::Plus, &Type::Int, &Type::String);
            assert_eq!(resolve_type_fresh(input), Err(expected.into()))
        }
    }

    mod cond_expr {
        use super::*;

        #[test]
        fn it_returns_error_for_non_bool_condition() {
            let input = make_tree("if (3) false else true;");
            let expected = InterpreterError::InvalidType { datatype: Type::Int };

            assert_eq!(resolve_type_fresh(input), Err(vec![expected]));
        }

        #[rstest]
        #[case::both_int("if (true) 3 else 4;", Type::Int)]
        #[case::both_string(r#"if (false) "a" else "b";"#, Type::String)]
        #[case::null_and_primitive(
            "if (true) null else 3;",
            Type::Int.as_nullable())
        ]
        #[case::disjoint_types(
            "if (false) 3 else true;",
            Type::Unknown
        )]
        #[case::list_and_empty_list(
            "if (true) [] else [123];",
            Type::Int.as_list()
        )]
        #[case::unknown_list(
            r#"if (false) [1] else ["string"];"#,
            Type::Unknown.as_list()
        )]
        #[case::nested_if_else_with_complex_type(
            "if (false) [] else if (true) null else [() -> 0];",
            Type::func(&[], Type::Int).as_list().as_nullable()
        )]
        fn it_infers_correct_type_for_both_conditions(
            #[case] input: &str, #[case] expected_type: Type
        ) {
            let syntax_tree = make_tree(input);
            let expected = ok_without_binding(expected_type);
            assert_eq!(resolve_type_fresh(syntax_tree), expected);
        }
    }

    mod func_call {
        use super::*;

        #[test]
        fn it_returns_error_for_undefined_function() {
            let input = make_tree("test(5);");
            let expected = vec![
                InterpreterError::UndefinedError { id: "test".into() }
            ];
            assert_eq!(resolve_type_fresh(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_non_function_id() {
            let tree = make_tree("five();");
            
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("five".into(), Type::Int);
            
            let err = InterpreterError::not_a_function(&Term::Id("five".into()));
            assert_eq!(resolve_type(&program, &tree), Err(vec![err]));
        }

        #[test]
        fn it_returns_error_for_wrong_num_args() {
            let tree = make_tree("f(1, 2, 3);");
            let program = make_program_with_func("f", &[Type::Int], Type::Int);

            let err = InterpreterError::ArgCount { got: 3, expected: 1 };
            assert_eq!(resolve_type(&program, &tree), Err(vec![err])); 
        }

        #[test]
        fn it_returns_error_for_mismatched_types() {
            let tree = make_tree(r#"f(false, "");"#);
            let program = make_program_with_func(
                "f", &[Type::Int, Type::Bool], Type::Int);

            let errs = vec![
                InterpreterError::UnexpectedType { got: Type::Bool, expected: Type::Int },
                InterpreterError::UnexpectedType { got: Type::String, expected: Type::Bool },
            ];
            assert_eq!(resolve_type(&program, &tree), Err(errs));
        }

        #[test]
        fn it_returns_ok_for_function_passed_in_with_less_args_than_param_type() {
            // passed in function only has one arg
            let tree = make_tree("f((arg: int) -> null);");
            // function definition has a param with a function with two args
            let param_type = Type::func(&[Type::Int, Type::Bool], Type::Null);
            let program =
                make_program_with_func("f", &[param_type], Type::Int);

            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_function_passed_in_without_declared_input_types() {
            let tree = make_tree("f((x) -> null);");
            let param_type = Type::func(&[Type::Int], Type::Null);
            let program =
                make_program_with_func("f", &[param_type], Type::String);

            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::String));
        }

        #[test]
        fn it_returns_ok_for_empty_defined_function() {
            let tree = make_tree("randInt();");
            let program = make_program_with_func("randInt",  &[], Type::Int);
            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_multi_arg_function() {
            let tree = make_tree("f(1, false);");
            let program = make_program_with_func(
                "f",  &[Type::Int, Type::Bool], Type::Int);
            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_infers_correct_return_type_from_generic_function_arguments() {
            let generic_t = || Type::Generic("T".into());
            let program = make_program_with_func(
                "f", &[generic_t()], generic_t());
            let tree = make_tree("f(false);",);
            assert_eq!(resolve_type(&program, &tree), ok_without_binding(Type::Bool));
        }

        /// Create a `Program` with a function of name `name`, input types
        /// `params`, and return type `return_type` in the type context
        fn make_program_with_func(
            name: &str,
            params: &[Type],
            return_type: Type
        ) -> Program {
            let mut program = Program::init_with_std_streams();
            let func_type = Type::func(params, return_type);
            program.type_context.insert(name.to_string(), func_type);
            program
        }
    }

    mod function {
        use crate::parser::test_utils::make_tree;

        use super::*;

        #[rstest]
        #[case::native_func("reverse;", &[Type::String], Type::String)]
        #[case::explicit_return_type("(): bool -> true;", &[], Type::Bool)]
        #[case::param_used_in_body("(x: int) -> x * x;", &[Type::Int], Type::Int)]
        #[case::returns_unknown("(): unknown -> 2025;", &[], Type::Unknown)]
        #[case::curried_function(
            // this function returns another function
            "(x: int) -> (y: int) -> x + y;",
            // input
            &[Type::Int],
            // return type
            Type::Func { input: vec![Type::Int], output: Box::new(Type::Int) }
        )]
        #[case::higher_order_func(
            "(f: (string -> int), x: string) -> f(x) + 3;",
            &[Type::func(&[Type::String], Type::Int), Type::String],
            Type::Int,
        )]
        fn it_returns_correct_function_type(
            #[case] input: &str,
            #[case] parameter_types: &[Type],
            #[case] return_type: Type,
        ) {
            let tree = make_tree(input);
            let expected = Type::Func {
                input: parameter_types.to_vec(),
                output: Box::new(return_type)
            };
            assert_eq!(resolve_type_fresh(tree), ok_without_binding(expected));
        }

        #[test]
        fn it_returns_ok_for_reused_variable_name_as_parameter() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".into(), Type::Bool);

            let input = make_tree("(x: null) -> x;");
            let result = resolve_type(&mut program, &input);

            assert_eq!(result, ok_without_binding(Type::Func {
                input: vec![Type::Null],
                output: Box::new(Type::Null),
            }));
        }

        #[test]
        fn it_returns_ok_for_explicit_generic_type() {
            let input = make_tree("<T>(x: T, y: int) -> x;");
            let result = resolve_type_fresh(input);
            assert_eq!(result, ok_without_binding(Type::func(
                &[generic_t(), Type::Int],
                generic_t()
            )))
        }

        #[test]
        fn it_returns_ok_for_generic_type_in_nested_function() {
            let input = make_tree("<T>() -> (x: T): T -> x;");
            let result = resolve_type_fresh(input);
            assert_eq!(result, ok_without_binding(Type::func(
                &[],
                Type::func(&[generic_t()], generic_t()))
            ));
        }

        #[rstest]
        #[case::undefined_symbol("(y: int) -> x;", &[error::undefined_id("x")])]
        #[case::bad_return_type(
            "(): bool -> \"\";",
            &[InterpreterError::bad_return_type(&Type::Bool, &Type::String)]
        )]
        #[case::reused_parameter(
            "(x: int) -> (y: bool) -> (x: int) -> x;",
            &[InterpreterError::ReassignError { id: "x".into() }]
        )]
        #[case::many_reused_parameters(
            "(x: int, y: int) -> (y: string, a: int, x: null) -> null;",
            &[
                InterpreterError::ReassignError { id: "y".into() },
                InterpreterError::ReassignError { id: "x".into() }
            ]
        )]
        #[case::recursive_function_without_return_type(
            "let f(x: int) -> f(x - 1);",
            &[InterpreterError::UndefinedError { id: "f".into() }]
        )]
        #[case::generic_param_not_in_type_param_list(
            "(x: T) -> null;",
            &[InterpreterError::UndeclaredGeneric { generic: "T".into() }]
        )]
        #[case::generic_return_type_not_in_type_param_list(
            "(): T -> null;",
            &[InterpreterError::UndeclaredGeneric { generic: "T".into() }]
        )]
        fn it_returns_error(#[case] input: &str, #[case] errors: &[InterpreterError]) {
            let tree = make_tree(input);
            assert_eq!(resolve_type_fresh(tree), Err(errors.to_vec()));
        }

        #[test]
        fn it_returns_ok_for_recursive_function() {
            let input = make_tree("let f(x: int): int -> f(x - 1);");
            let func_type = Type::Func {
                input: vec![Type::Int],
                output: Box::new(Type::Int)
            };
            assert_eq!(resolve_type_fresh(input), ok_with_binding("f", func_type));
        }

        /// shorthand for a generic type 'T'
        fn generic_t() -> Type {
            Type::Generic("T".into())
        }
    }
}
