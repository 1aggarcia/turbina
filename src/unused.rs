// First attempt to create an interpreter with poor data abstractions
// This might be useful later when I try to evaluate ASTs

struct Program {
    vars: HashMap<String, LiteralValue>
}

#[derive(Clone, Debug)]
enum LiteralValue {
    INT(i32),
    BOOL(bool),
    STRING(String),
    NULL,
}


fn evaluate(tokens: Vec<String>, program: &mut Program) -> Result<LiteralValue, String> {
    if tokens.is_empty() {
       panic!("Tried to evaluate empty tokens");
    }
    let keyword = &tokens[0];
    if keyword == "let" {
        eval_let(tokens, program)
    } else if tokens.len() > 1 {
        eval_math(tokens, program)
    } else {
        eval_token(&tokens[0], program)
    }
}

fn eval_let(tokens: Vec<String>, program: &mut Program) -> Result<LiteralValue, String> {
    let keyword = &tokens[0];
    if keyword != "let" {
        return Err("Invalid keyword".to_string());
    }
    // parse as "let" operation
    if tokens.len() < 4 {
        return Err("Not enough arugments".to_string());
    }
    let var_name = &tokens[1];
    if !var_name.chars().all(|c| c.is_alphabetic()) {
        return Err("Variable name must be only alphabetical characters".to_string())
    }
    if program.vars.contains_key(var_name) {
        return Err("cannot redefine variable".to_string());
    }
    if tokens[2] != "=" {
        return Err("Invalid syntax: expected '='".to_string());
    }
    let expression = tokens[3..].to_vec();
    let value_result = evaluate(expression, program);
    if value_result.is_err() {
        return value_result
    }
    let value = value_result.unwrap();
    program.vars.insert(var_name.to_string(), value.clone());
    println!("{:?}", program.vars);

    return Ok(value);
}

fn eval_math(tokens: Vec<String>, program: &mut Program) -> Result<LiteralValue, String> {
    if tokens.len() != 3 {
        return Err("Syntax error: wrong number of arguments".to_string());
    }
    let operator = &tokens[1];

    // type checking
    let arg0 = evaluate(tokens[0..1].to_vec(), program)?;
    let parsed0 = match arg0 {
        LiteralValue::INT(i) => i,
        _ => return Err("Type error: arg0 is not int".to_string()),
    };

    let arg1 = evaluate(tokens[2..3].to_vec(), program)?;
    let parsed1 = match arg1 {
        LiteralValue::INT(i) => i,
        _ => return Err("Type error: arg1 is not int".to_string()),
    };

    // evaluate
    let result = match operator.as_str() {
        "+" => parsed0 + parsed1,
        "-" => parsed0 - parsed1,
        "*" => parsed0 * parsed1,
        "/" => parsed0 / parsed1,
        "%" => parsed0 % parsed1,
        _ => return Err("Syntax error: unknown operator".to_string())
    };

    Ok(LiteralValue::INT(result))
}

fn eval_token(token: &str, state: &Program) -> Result<LiteralValue, String> {
    if token == "null" {
        return Ok(LiteralValue::NULL);
    }
    if token == "true" {
        return Ok(LiteralValue::BOOL(true));
    }
    if token == "false" {
        return Ok(LiteralValue::BOOL(false));
    }
    if state.vars.contains_key(token) {
        return Ok(state.vars.get(token).unwrap().clone());
    }
    if token.starts_with("\"") && token.ends_with("\"") {
        return Ok(LiteralValue::STRING(token[1..token.len() - 1].to_string()));
    }
    match token.parse::<i32>() {
        Err(_) => Err("invalid token".to_string()),
        Ok(i) => Ok(LiteralValue::INT(i)),
    }
}

fn value_to_string(value: LiteralValue) -> String {
    match value {
        LiteralValue::INT(i) => i.to_string(),
        LiteralValue::STRING(s) => s,
        LiteralValue::BOOL(b) => (
            if b { "true" }
            else { "false" }
        ).to_string(),
        LiteralValue::NULL => "null".to_string(),
    }
}