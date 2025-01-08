Variables and math operators
```
// variables are immutable
let x = 4   // 4
x = 5       // ERROR

let a = 2 + 3   // 5
let b = 2 * 3   // 6

let c = x - a   // -1

let d = 7 / 2   // 3
let e = 7 % 2   // 1
```

References
```
// References point to mutable data
ref x = 4   // Ref(4)
x.set(5)    // Ref(5)

// Reading a reference extracts the data
let z = x       // Ref(5)
let y = x.get() // 5
```

Control flow and boolean operators
```
// 1
if (true) 1 else 0

let num = 15

// "odd"
let x = if (num % 2 == 0) {
    "even"
} else {
    "odd"
}

ref i = 0
ref sum = 0
while (i < 15) {
    sum.set(sum + i)
    i.set(i + 1)
}
```

Functions
```
let square (x: int) = x * x
let four = square(2)        // 4

let factorial (x: int) = {
    if (x == 0) {
        return x
    }
    return x * factorial(x - 1)
}

let evens = [0, 1, 2, 3, 4].filter((x) -> x % 2 == 0)
```

Lists
```
let nums = [1, 2, 3, 4, 5]
let nested = [[0, 3], [4, 3], [6, 2]]
let illegal = [true, 2]     // ERROR

nums.map(square)            // [2, 4, 9, 16, 25]
nums.filter(isEven)         // [2, 4]
nums.reduce(sum)            // 15
```
