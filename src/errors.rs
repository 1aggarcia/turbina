// Shorthand for formatted Err results, prefixed by the error type

macro_rules! syntax_error {
    ($($arg:tt)*) => {
        Err("Syntax Error: ".to_string() + format!($($arg)*).as_str())
    }
}

pub(crate) use syntax_error;
