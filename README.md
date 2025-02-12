# Turbina

Turbina is an interpreted, statically typed, functional programming language whose interpreter is written in [Rust](https://www.rust-lang.org/). It takes inspiration from OCaml, Rust and TypeScript. [Try Turbina online here](https://turbinalang.web.app/).

Turbina supports execution via both command-line interface and source code files.

- To execute a source file run `cargo run <filepath>` from the root directory.
- To run the REPL (read-eval-print loop), run `cargo run` without a filepath.

Below is a draft of the syntax. A formal grammar is in the works.

**Variables**
```rust
let x: int = 4;   // 4
let x: int = 5;   // ERROR - variables are immutable
x = 5;            // also an error

let y = "a string";  // type is inferred
```

**Datatypes**
```rust
// primitives
let a: int = 1;
let b: bool = true;
let c: string = "a string";

// lists
let d: int[] = [1, 2, 3];
let e: bool[] = [true, false];

// functions
type predicate = (int) -> bool;  // int arg, returns bool
type producer = () -> string;  // no arg, returns string
type consumer = (int, string) -> null;  // two args, returns null

/// nullable types
let nullableString: string? = "some string";
let anotherNullableString: string? = null;

// string? might be null, so it's not allowed to be assigned to string
let notNullString: string = nullableString;     // ERROR

// However, you can assert it as not-null with the "!" operator.
let notNullString: string = nullableString!;    // OK

// nullable functions types are wrapped in parentheses to avoid ambiguity
type nullableFunction = ((int) -> int)?
```

**Functions**
```rust
// inferred return type
let concat = (a: string, b: string) -> a + b

// the "=" can be elimnated for function bindings
let concat2(a: string, b: string) -> a + b

// declared return type
let factorial(x: int): int {
    if (x == 0) {
        return 1;
    }
    return x * factorial(x - 1);
}

// alternatively
let factorial2(x: int) ->
    if (x == 0) 1
    else x * factorial2(x - 1);

// else if
let isEven(n: int) ->
    if (n == 0) true
    else if (n == 1) false
    else if (n == 2) true
    else if (n == 3) false
    else if (n == 4) true
    ...
```

**Higher-order functions**
```rust
// function as an argument (also showcases anonymous functions)
let squares = map([1, 2, 3, 4], (x: int) -> x * x);

// define the function like this
let map(nums: int[], func: ((int) -> int)) {
    ...
}

// function as a return value
let add(x: int) -> (y: int) -> x + y;
```
