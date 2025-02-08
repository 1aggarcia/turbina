use std::collections::HashMap;

use crate::models::{AbstractSyntaxTree, BinaryExpr, BinaryOp, Closure, CondExpr, EvalContext, Expr, FuncBody, FuncCall, Function, LetNode, Literal, Program, Scope, Term};

/// Execute the statement represented by the AST on the program passed in.
/// 
/// Syntax errors will cause a panic and should be checked with the
/// `validate` function first.
pub fn evaluate(program: &mut Program, tree: &AbstractSyntaxTree) -> Literal {
    // TODO: use Result type for runtime errors
    let mut global_context = EvalContext {
        output: &mut program.output,
        scope: Scope {
            bindings: &mut program.bindings,
            parent: None,
        }
    };

    match tree {
        AbstractSyntaxTree::Let(node) => eval_let(&mut global_context, node),
        AbstractSyntaxTree::Expr(node) => eval_expr(&mut global_context, node),
    }
}

/// Bind an expression to a name
fn eval_let(context: &mut EvalContext, node: &LetNode) -> Literal {
    if context.scope.bindings.contains_key(&node.id) {
        panic!("variable '{}' already defined", node.id);
    }
    let literal_value = eval_expr(context, &node.value);
    context.scope.bindings.insert(node.id.clone(), literal_value.clone());

    return literal_value;
}

fn eval_expr(context: &mut EvalContext, expr: &Expr) -> Literal {
    match expr {
        Expr::Binary(b) => eval_binary_expr(context, b),
        Expr::Cond(c) => eval_cond_expr(context, c),
        Expr::Function(f) => eval_function(context, f),
    }
}

/// Reduce a sequence of terms and operators to a single literal
fn eval_binary_expr(context: &mut EvalContext, expr: &BinaryExpr) -> Literal {
    let mut result = eval_term(context, &expr.first);

    for (op, term) in &expr.rest {
        let new_arg = eval_term(context, &term);
        result = eval_binary_op(result, op, new_arg);
    }

    return result;
}

fn eval_cond_expr(context: &mut EvalContext, expr: &CondExpr) -> Literal {
    let cond_result = eval_expr(context, &expr.cond);
    match cond_result {
        Literal::Bool(true) => eval_expr(context, &expr.if_true),
        Literal::Bool(false) => eval_expr(context, &expr.if_false),
        _ => panic!("TYPE CHECKER FAILED: condition did not evaluate to bool: {cond_result:?}")
    }
}

/// Returns a closure with the passed in function and parent scope saved to it
/// (if it is not the global sopce)
fn eval_function(context: &mut EvalContext, function: &Function) -> Literal {
    let parent_scope = if context.scope.is_local_scope() {
        context.scope.bindings.clone()
    } else {
        HashMap::new()
    };

    let closure = Closure { function: function.clone(), parent_scope };
    Literal::Closure(closure)
}

/// Invoke a function with the passed in arguments
fn eval_func_call(context: &mut EvalContext, call: &FuncCall) -> Literal {
    let Literal::Closure(closure) = eval_term(context, &call.func) else {
        panic!("bad type: {:?}", call.func);
    };
    let Closure { function, parent_scope } = closure;

    let args: Vec<Literal> = call.args.iter()
        .map(|a| eval_expr(context, a)).collect();

    let function_body = match function.body {
        FuncBody::Native(native_func) => return native_func(args, context),
        FuncBody::Expr(expr) => *expr,
    };

    // create local scope with arguments bound to parameters
    let mut function_bindings = parent_scope;
    for ((param_name, _), arg) in function.params.iter().zip(args.iter()) {
        function_bindings.insert(param_name.clone(), arg.clone());
    }

    let mut function_context = EvalContext {
        output: context.output,
        scope: Scope {
            bindings: &mut function_bindings,
            parent: Some(&context.scope),
        }
    };

    eval_expr(&mut function_context, &function_body)
}

fn eval_term(context: &mut EvalContext, term: &Term) -> Literal {
    #[inline(always)]
    fn eval_negated(context: &mut EvalContext, inner_term: &Term) -> Literal {
        let inner_result = eval_term(context, inner_term);
        match inner_result {
            Literal::Bool(bool) => Literal::Bool(!bool),
            Literal::Int(int) => Literal::Int(-int),
            _ => panic!("expected bool or int, got {:?}", inner_result)
        }
    }

    match term {
        Term::Literal(lit) => lit.clone(),
        Term::Id(id) => eval_id(context, id),
        Term::Not(t) | Term::Minus(t) => eval_negated(context, t),
        Term::Expr(expr) => eval_expr(context, expr),
        Term::FuncCall(call) => eval_func_call(context, call),
        Term::NotNull(term) => eval_term(context, term),
    }
}

/// Lookup the value stored for a variable name
fn eval_id(context: &mut EvalContext, id: &str) -> Literal {
    match context.scope.lookup(id) {
        Some(literal) => literal.clone(),

        // TODO: this should never happen, but use result type anyway
        None => panic!("variable '{}' does not exist", id),
    }
}

/// Helper to compute the result of the binary operation
fn eval_binary_op(left: Literal, operator: &BinaryOp, right: Literal) -> Literal {
    match operator {
        BinaryOp::Plus => eval_plus(left, right),
        BinaryOp::Minus =>
            Literal::Int(literal_as_int(left) - literal_as_int(right)),
        BinaryOp::Star =>
            Literal::Int(literal_as_int(left) * literal_as_int(right)),
        BinaryOp::Slash =>
            Literal::Int(literal_as_int(left) / literal_as_int(right)),
        BinaryOp::Percent =>
            Literal::Int(literal_as_int(left) % literal_as_int(right)),

        // these use the derived `PartialEq` trait on enum `Literal`
        BinaryOp::Equals => Literal::Bool(left == right),
        BinaryOp::NotEq => Literal::Bool(left != right),
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
            str + &literal_to_string(right).unwrap() 
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

/// Tries to extract a string from a literal, if possible
fn literal_to_string(literal: Literal) -> Option<String> {
    match literal {
        Literal::String(str) => Some(str),
        _ => None,
    }
}

#[cfg(test)]
mod test_evalutate {
    use crate::{models::test_utils::term_tree, parser::test_utils::make_tree};

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
        assert_eq!(evaluate(&mut program, &input), Literal::Bool(true));
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

        evaluate(&mut program, &tree);
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
        assert_eq!(evaluate(&mut program, &input), Literal::Null);
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
    // TODO: thorough PEMDAS test
    fn it_evaluates_complex_expressions(
        #[case] input: &str, #[case] expected_val: i32
    ) {
        let input = make_tree(input);
        let expected = Literal::Int(expected_val);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[rstest]
    // ints
    #[case("3 == 5;", 3 == 5)]
    #[case("12 == 12;", 12 == 12)]
    #[case("0 != 0;", 0 != 0)]
    #[case("2 != 1;", 2 != 1)]

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
    fn it_evaluates_binary_bool_operators(
        #[case] input: &str, #[case] expected_val: bool
    ) {
        let input = make_tree(input);
        let expected = Literal::Bool(expected_val);
        assert_eq!(evaluate_fresh(input), expected);
    }

    #[test]
    fn it_evaluates_string_concatenation() {
        let input = make_tree("\"abc\" + \"xyz\";");
        let expected = Literal::String("abcxyz".to_string());
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

        evaluate(&mut program, &definition);
        assert_eq!(evaluate(&mut program, &invocation), Literal::Int(25));
    }

    #[test]
    fn it_calls_function_with_params_overriding_global_scope() {
        let define_x = make_tree(r#"let x = "global string value";"#);
        let define_square = make_tree("let square = (x: int) -> x * x;");
        // the function call should use the int argument for x and not the
        // global string
        let square_three = make_tree("square(3);");
        let mut program = Program::init_with_std_streams();

        evaluate(&mut program, &define_x);
        evaluate(&mut program, &define_square);
        assert_eq!(evaluate(&mut program, &square_three), Literal::Int(9));
    }

    // TODO: fix by adding closures to function literals
    #[test]
    fn it_saves_scope_for_curried_function() {
        let mut program = Program::init_with_std_streams();

        let define_sum = "let sum = (a:int)->(b:int)->(c:int)->a+b+c;";
        evaluate(&mut program, &make_tree(define_sum));
        evaluate(&mut program, &make_tree("let addThree = sum(3);"));
        evaluate(&mut program, &make_tree("let addTen = addThree(7);"));

        assert_eq!(evaluate(&mut program, &make_tree("addTen(2015);")), Literal::Int(2025));
    }

    // evaluate an AST on a new program
    fn evaluate_fresh(tree: AbstractSyntaxTree) -> Literal{
        return evaluate(&mut Program::init_with_std_streams(), &tree);
    }
}
