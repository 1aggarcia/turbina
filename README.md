# Turbina

Turbina is an interpreted, statically typed, functional programming language whose interpreter is written in [Rust](https://www.rust-lang.org/). It takes inspiration from OCaml and TypeScript.

Turbina will support execution via both command-line interface and source code files.

To run the REPL (read-eval-print loop), run `cargo run` from the root directory.

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
type consumer = (int, string) -> void;  // two args, returns nothing
```

**Functions**
```rust
// inferred return type
let concat = (a: string, b: string) -> a + b

// declared return type
let factorial = (x: int): int {
    if (x == 0) {
        return 1
    }
    return x * factorial(x - 1)
}

// alternatively
let other_fac = (x: int) -> if (x == 0) 1 else x * factorial(x - 1)
```

**Higher-order functions**
```rust
// function as an argument
let printOutput = (input: int, func: (int) -> int) {
    print(func(input))
}

// function as a return value
let add = (x: int) -> (y: int) -> x + y;
```
