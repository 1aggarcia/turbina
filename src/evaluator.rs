use std::collections::HashMap;

use crate::{errors::Result, models::{AbstractSyntaxTree, BinaryExpr, BinaryOp, Closure, CodeBlock, CondExpr, EvalContext, Expr, FuncBody, FuncCall, Function, LetNode, Literal, Program, Scope, Term}};

/// Execute the statement represented by the AST on the program passed in.
/// 
/// Syntax errors will cause a panic and should be checked with the
/// `validate` function first.
pub fn evaluate(program: &mut Program, tree: &AbstractSyntaxTree) -> Result<Literal> {
    let mut global_context = EvalContext {
        output: &mut program.output,
        scope: Scope {
            bindings: &mut program.bindings,
            parent: None,
        }
    };
    // TODO: remove reference, this should be able to consume the tree
    let owned_tree: AbstractSyntaxTree = tree.clone();

    // TODO: integrate result type into helpers below for runtime errors
    Ok(eval_statement(&mut global_context, owned_tree))
}

fn eval_statement(context: &mut EvalContext, statement: AbstractSyntaxTree) -> Literal {
    match statement {
        AbstractSyntaxTree::Let(node) => eval_let(context, node),
        AbstractSyntaxTree::Expr(node) => eval_expr(context, node),
    }
}

/// Bind an expression to a name
fn eval_let(context: &mut EvalContext, node: LetNode) -> Literal {
    if context.scope.bindings.contains_key(&node.id) {
        panic!("variable '{}' already defined", node.id);
    }
    let literal_value = eval_expr(context, node.value);
    context.scope.bindings.insert(node.id, literal_value.clone());

    return literal_value;
}

fn eval_expr(context: &mut EvalContext, expr: Expr) -> Literal {
    match expr {
        Expr::Binary(b) => eval_binary_expr(context, b),
        Expr::CodeBlock(b) => eval_code_block(context, b),
        Expr::Cond(c) => eval_cond_expr(context, c),
        Expr::Function(f) => eval_function(context, f),
    }
}

/// Reduce a sequence of terms and operators to a single literal
fn eval_binary_expr(context: &mut EvalContext, expr: BinaryExpr) -> Literal {
    let mut result = eval_term(context, expr.first);

    for (op, term) in expr.rest {
        if op == BinaryOp::Pipe {
            // create a local variable under "_" to pipe into the right side
            context.scope.bindings.insert("_".into(), result.clone());
        }
        let new_arg = eval_term(context, term);
        // cleanup temp variable so that it isn't leaked
        if op == BinaryOp::Pipe {
            context.scope.bindings.remove("_");
        }
        result = eval_binary_op(result, &op, new_arg);
    }

    return result;
}

fn eval_code_block(context: &mut EvalContext, block: CodeBlock) -> Literal {
    let mut local_scope = if context.scope.is_local_scope() {
        // clone the current scope so that closures can use it
        // TODO: rethink closures and eliminate need to cloning
        context.scope.bindings.clone()
    } else {
        HashMap::new()
    };
    let mut local_context = EvalContext {
        output: context.output,
        scope: Scope {
            bindings: &mut local_scope,
            parent: Some(&context.scope)
        }
    };
    block.statements
        .into_iter()
        .fold(Literal::Null, |_, statement|
            eval_statement(&mut local_context, statement)
        )
}

fn eval_cond_expr(context: &mut EvalContext, expr: CondExpr) -> Literal {
    let cond_result = eval_expr(context, *expr.cond);
    match cond_result {
        Literal::Bool(true) => eval_expr(context, *expr.if_true),
        Literal::Bool(false) => eval_expr(context, *expr.if_false),
        _ => panic!("TYPE CHECKER FAILED: condition did not evaluate to bool: {cond_result:?}")
    }
}

/// Returns a closure with the passed in function and parent scope saved to it
/// (if it is not the global scope)
fn eval_function(context: &mut EvalContext, function: Function) -> Literal {
    let parent_scope = if context.scope.is_local_scope() {
        context.scope.bindings.clone()
    } else {
        HashMap::new()
    };

    let closure = Closure { function: function, parent_scope };
    Literal::Closure(closure)
}

/// Invoke a function with the passed in arguments.
/// Made public so that library functions (e.g. map, filter, reduce) can
/// evaluate user-supplied functions.
pub fn eval_func_call(
    context: &mut EvalContext,
    closure: Closure, 
    args: Vec<Literal>,
) -> Literal {
    let Closure { function, mut parent_scope } = closure;
    let function_body = match function.body {
        FuncBody::Native(native_func) => {
            let mut function_context = EvalContext {
                output: context.output,
                scope: Scope {
                    bindings: &mut parent_scope,
                    parent: Some(&context.scope.get_global_scope()),
                }
            };
            return native_func(args, &mut function_context);
        },
        FuncBody::Expr(expr) => expr,
    };

    // create local scope with arguments bound to parameters
    for ((param_name, _), arg) in function.params.iter().zip(args.into_iter()) {
        parent_scope.insert(param_name.clone(), arg);
    }

    let mut function_context = EvalContext {
        output: context.output,
        scope: Scope {
            bindings: &mut parent_scope,
            parent: Some(&context.scope.get_global_scope()),
        }
    };
    eval_expr(&mut function_context, *function_body)
}

/// Evaluates all expressions in the list passed in
fn eval_list(context: &mut EvalContext, list: Vec<Expr>) -> Literal {
    let literals = list
        .into_iter()
        .map(|expr| eval_expr(context, expr))
        .collect();
    Literal::List(literals)
}

fn eval_term(context: &mut EvalContext, term: Term) -> Literal {
    #[inline(always)]
    fn eval_negated(context: &mut EvalContext, inner_term: Term) -> Literal {
        let inner_result = eval_term(context, inner_term);
        match inner_result {
            Literal::Bool(bool) => Literal::Bool(!bool),
            Literal::Int(int) => Literal::Int(-int),
            _ => panic!("expected bool or int, got {:?}", inner_result)
        }
    }

    #[inline(always)]
    fn eval_wrapped_func_call(context: &mut EvalContext, call: FuncCall) -> Literal {
        let Literal::Closure(closure) = eval_term(context, *call.func.clone()) else {
            panic!("bad type: {:?}", call.func);
        };
        let args: Vec<Literal> = call.args.into_iter()
            .map(|a| eval_expr(context, a)).collect();
        eval_func_call(context, closure, args)
    }

    match term {
        Term::Literal(lit) => lit,
        Term::Id(id) => eval_id(context, &id),
        Term::Not(t) | Term::Minus(t) => eval_negated(context, *t),
        Term::Expr(expr) => eval_expr(context, *expr),
        Term::FuncCall(call) => eval_wrapped_func_call(context, call),
        Term::NotNull(term) => eval_term(context, *term),
        Term::List(list) => eval_list(context, list),
    }
}

/// Lookup the value stored for a variable name
fn eval_id(context: &mut EvalContext, id: &str) -> Literal {
    match context.scope.lookup(id) {
        Some(literal) => literal,

        // TODO: this should never happen, but use result type anyway
        None => panic!("variable '{}' does not exist: context {:#?}", id, context),
    }
}

/// Helper to compute the result of the binary operation
fn eval_binary_op(left: Literal, operator: &BinaryOp, right: Literal) -> Literal {
    use BinaryOp::*;

    match operator {
        Plus => eval_plus(left, right),
        Minus =>
            Literal::Int(literal_as_int(left) - literal_as_int(right)),
        Star =>
            Literal::Int(literal_as_int(left) * literal_as_int(right)),
        Slash =>
            Literal::Int(literal_as_int(left) / literal_as_int(right)),
        Percent =>
            Literal::Int(literal_as_int(left) % literal_as_int(right)),

        GreaterThan =>
            Literal::Bool(literal_as_int(left) > literal_as_int(right)),
        GreaterThanOrEqual =>
            Literal::Bool(literal_as_int(left) >= literal_as_int(right)),
        LessThan =>
            Literal::Bool(literal_as_int(left) < literal_as_int(right)),
        LessThanOrEqual =>
            Literal::Bool(literal_as_int(left) <= literal_as_int(right)),

        And => Literal::Bool(literal_as_bool(left) && literal_as_bool(right)),
        Or => Literal::Bool(literal_as_bool(left) || literal_as_bool(right)),

        // these use the derived `PartialEq` trait on enum `Literal`
        Equals => Literal::Bool(left == right),
        NotEq => Literal::Bool(left != right),

        Pipe => right,
    }
}

/// perform either string concatenation or integer addition,
/// depending on the literal type
fn eval_plus(left: Literal, right: Literal) -> Literal {
    let left_as_str = literal_to_string(left.clone());
    match left_as_str {
        None => Literal::Int(
            literal_as_int(left) + literal_as_int(right)
        ),
        Some(str) => Literal::String(
            str + &literal_to_string(right.clone()).expect(
                &format!("right side of + was a non-string literal: {right}"))
        ),
    }
}

/// Casts a literal to an int
fn literal_as_int(literal: Literal) -> i32 {
    match literal {
        Literal::Int(i) => i,
        _ => panic!("expected int literal, got {:?}", literal),
    }
}

fn literal_as_bool(literal: Literal) -> bool {
    match literal {
        Literal::Bool(b) => b,
        _ => panic!("expected bool literal, got {:?}", literal),
    }
}

/// Tries to extract a string from a literal, if possible
fn literal_to_string(literal: Literal) -> Option<String> {
    match literal {
        Literal::String(str) => Some(str),
        _ => None,
    }
}

#[cfg(test)]
mod test_evalutate {
    use crate::{models::test_utils::*, parser::test_utils::make_tree};

    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(Literal::Bool(true))]
    #[case(Literal::Int(5))]
    #[case(Literal::String("so many tesssssts".to_string()))]
    fn it_returns_input_on_literals(#[case] literal: Literal) {
        let input = term_tree(Term::Literal(literal.clone()));
        assert_eq!(evaluate_fresh(input), literal);
    }

    #[rstest]
    #[case("!false;", Literal::Bool(true))]
    #[case("!true;", Literal::Bool(false))]
    #[case("-15;", Literal::Int(-15))]
    fn it_negates_literals(#[case] input: &str, #[case] expected: Literal) {
        let tree = make_tree(input);
        assert_eq!(evaluate_fresh(tree), expected);
    }

    #[test]
    fn it_looks_up_variables_correctly() {
        let mut program = Program::init_with_std_streams();
        program.bindings.insert("is_lang_good".to_string(), Literal::Bool(true));
        program.bindings.insert("some_int".to_string(), Literal::Int(-5));

        let input = make_tree("is_lang_good;");
        assert_eq!(force_evaluate(&mut program, &input), Literal::Bool(true));
    }

    #[rstest]
    #[case("let t = true;", Literal::Bool(true))]
    #[case("let t: null = null;", Literal::Null)]
    fn it_binds_literal_value_to_symbol(
        #[case] input: &str,
        #[case] value: Literal,
    ) {
        let mut program = Program::init_with_std_streams();
        let tree = make_tree(input);

        force_evaluate(&mut program, &tree);
        assert_eq!(program.bindings["t"], value);
    }

    #[test]
    fn it_returns_value_of_var_after_binding() {
        let input = make_tree("let t = 12345 - 98765;");
        let expected = Literal::Int(12345 - 98765);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_binds_casted_nullable_value() {
        let mut program = Program::init_with_std_streams();
        program.bindings.insert("nullString".into(), Literal::Null);

        let input = make_tree("let validString: string = nullString!;");
        assert_eq!(force_evaluate(&mut program, &input), Literal::Null);
    }

    #[rstest]
    #[case("3 - 5;", 3 - 5)]
    #[case("3 + 5;", 3 + 5)]
    #[case("3 * 5;", 3 * 5)]
    #[case("3 / 5;", 3 / 5)]
    #[case("3 % 5;", 3 % 5)]
    fn it_evaluates_binary_math_operators(
        #[case] input: &str, #[case] expected_val: i32
    ) {
        let input = make_tree(input);
        let expected = Literal::Int(expected_val);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[rstest]
    #[case("3 * 2 - 5;", 1)]
    #[case("3 * (2 - 5);", -9)]
    #[case("(25 % (18 / 3)) - (10 + 4);", -13)]
    #[case("(25 % 18 / 3) - (10 + 4);", -12)]
    #[case::pemdas(
        "1 + 2 * 3 - 4 / 5 + 6 % 7;",
        1 + (2 * 3) - (4 / 5) + (6 % 7)
    )]
    #[case::function_piping(
        // start with boolean array
        // negate all booleans -> [true, false, true]
        // filter out false -> [true, true]
        // join to string -> "truetrue"
        // get string length -> 8
        r#"
        [false, true, false]
            |> map(_, (val: bool) -> !val)
            |> filter(_, (val: bool) -> val == true)
            |> join(_, "")
            |> len(_);
        "#,
        8
    )]
    fn it_evaluates_complex_expressions(
        #[case] input: &str, #[case] expected_val: i32
    ) {
        let input = make_tree(input);
        let expected = Literal::Int(expected_val);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[rstest]
    // int equality
    #[case("3 == 5;", 3 == 5)]
    #[case("12 == 12;", 12 == 12)]
    #[case("0 != 0;", 0 != 0)]
    #[case("2 != 1;", 2 != 1)]

    // int comparison
    #[case("2 > 1;", 2 > 1)]
    #[case("2 < 1;", 2 < 1)]
    #[case("2 >= 2;", 2 >= 2)]
    #[case("2 <= 2;", 2 <= 2)]

    // strings
    #[case("\"a\" == \"a\";", true)]
    #[case("\"a\" != \"a\";", false)]
    #[case("\"abc\" == \"efg\";", false)]
    #[case("\"efg\" != \"abc\";", true)]

    // bools
    #[case("true == true;", true)]
    #[case("true != true;", false)]
    #[case("false == true;", false)]
    #[case("true != false;", true)]

    #[case("true && true;", true)]
    #[case("true && false;", false)]
    #[case("false || false;", false)]
    #[case("false || true;", true)]

    #[case("false || (true && false);", false)]
    #[case("(false && true) || (false || true);", true)]

    #[case("4 - 3 > 3 - 4;", true)]
    #[case("4 - 3 > 3 - 4 && 4 - 3 < 3 - 4;", false)]
    fn it_evaluates_binary_bool_operators(
        #[case] input: &str, #[case] expected_val: bool
    ) {
        let input = make_tree(input);
        let expected = Literal::Bool(expected_val);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_evaluates_list_of_literals() {
        let input = make_tree("[1, 2, 3];");
        let expected = Literal::List(
            vec![Literal::Int(1), Literal::Int(2), Literal::Int(3)]
        );
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_evaluates_list_of_expressions() {
        let input = make_tree("[false && true, (() -> 15)()];");
        let expected = Literal::List(
            vec![Literal::Bool(false), Literal::Int(15)]
        );
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_evaluates_string_concatenation() {
        let input = make_tree("\"abc\" + \"xyz\";");
        let expected = Literal::String("abcxyz".to_string());
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_evaluates_code_blocks() {
        let input = make_tree(r#"{
            let string_a = if (true) {
                "a" + "A"
            } else {
                "ERROR"
            };
            let string_b = if (false) {
                "b" + "B"
            } else {
                "ERROR"
            };
            string_a + " " + string_b
        };"#);
        let expected = Literal::String("aA ERROR".into());
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_captures_closure_environment_correctly_in_global_scope() {
        let mut program = Program::init_with_std_streams();

        let define_y = make_tree("let y = 5;");
        let define_closure = make_tree("let closure = () -> y;");
        let define_override_y = make_tree(
            "let overrideY = (y: string) -> closure();");

        force_evaluate(&mut program, &define_y);
        force_evaluate(&mut program, &define_closure);
        force_evaluate(&mut program, &define_override_y);

        let call_override_y = make_tree(r#"overrideY("should not be returned");"#);
        let expected = Literal::Int(5);
        assert_eq!(force_evaluate(&mut program, &call_override_y), expected);
    }

    #[test]
    fn it_captures_closure_environment_correctly_in_code_blocks() {
        let input = make_tree(r#"{
            let y: string = "not an int";
            let overrideY = (y: int) -> { () -> y * 2 };
            overrideY(5)()
        };"#);
        let expected = Literal::Int(10);

        assert_eq!(evaluate_fresh(input), expected);
    }

    #[rstest]
    #[case("if (2 == 3) 1 else 2;", 2)]
    #[case("if (2 != 3) 3 else 4;", 3)]
    fn it_evaluates_conditional_expression(#[case] input: &str, #[case] expected: i32) {
        let input = make_tree(input);
        assert_eq!(evaluate_fresh(input), Literal::Int(expected));
    }

    #[test]
    fn it_calls_native_library_function() {
        let input = make_tree(r#"reverse("12345");"#);
        let expected = Literal::String("54321".into());
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_calls_user_defined_function() {
        let definition = make_tree("let square = (x: int) -> x * x;");
        let invocation = make_tree("square(5);");
        let mut program = Program::init_with_std_streams();

        force_evaluate(&mut program, &definition);
        assert_eq!(force_evaluate(&mut program, &invocation), Literal::Int(25));
    }

    #[test]
    fn it_calls_function_with_params_overriding_global_scope() {
        let define_x = make_tree(r#"let x = "global string value";"#);
        let define_square = make_tree("let square = (x: int) -> x * x;");
        // the function call should use the int argument for x and not the
        // global string
        let square_three = make_tree("square(3);");
        let mut program = Program::init_with_std_streams();

        force_evaluate(&mut program, &define_x);
        force_evaluate(&mut program, &define_square);
        assert_eq!(force_evaluate(&mut program, &square_three), Literal::Int(9));
    }

    #[test]
    fn it_saves_scope_for_curried_function() {
        let mut program = Program::init_with_std_streams();

        let define_sum = "let sum = (a:int)->(b:int)->(c:int)->a+b+c;";
        force_evaluate(&mut program, &make_tree(define_sum));
        force_evaluate(&mut program, &make_tree("let addThree = sum(3);"));
        force_evaluate(&mut program, &make_tree("let addTen = addThree(7);"));

        assert_eq!(
            force_evaluate(&mut program, &make_tree("addTen(2015);")),
            Literal::Int(2025)
        );
    }

    #[test]
    fn it_calls_function_with_function_argument() {
        let mut program = Program::init_with_std_streams();

        force_evaluate(&mut program, &make_tree(
            "let compose(f: (int -> int), g: (int -> int)) -> (x: int) -> f(g(x));"));
            force_evaluate(&mut program, &make_tree(
            "let addThree = compose((x: int) -> x + 1, (x: int) -> x + 2);"));
        assert_eq!(
            force_evaluate(&mut program, &make_tree("addThree(5);")),
            Literal::Int(8)
        );
    }

    #[test]
    fn it_calls_passed_in_function_with_less_args_than_param_type_defines() {
        let mut program = Program::init_with_std_streams();

        force_evaluate(&mut program, &make_tree(
            "let callBinaryFunc(binaryFunc: ((int, int) -> int)) -> binaryFunc(4, 14);"));
        let result = force_evaluate(&mut program, &make_tree(
            "callBinaryFunc((x: int) -> x * 2);"));
        assert_eq!(result, Literal::Int(8));
    }

    // evaluate an AST on a new program
    fn evaluate_fresh(tree: AbstractSyntaxTree) -> Literal {
        evaluate(&mut Program::init_with_std_streams(), &tree).unwrap()
    }

    fn force_evaluate(program: &mut Program, tree: &AbstractSyntaxTree) -> Literal {
        evaluate(program, &tree).unwrap()
    }
}
