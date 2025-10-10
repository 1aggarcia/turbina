const BASIC_SAMPLE_PROGRAM =
`// define a binding like this
let x: int = 5;

// the type can be inferred
let y = "this is a string";

// 'let' bindings cannot be reassigned
// let x = 20;  // ERROR

// define functions like this
let square(x: int) -> x * x;
println(square(5));

// or do it the long way
let otherSquare = (x: int) -> x * x;

// all if statements are expressions
let isPalindrome(str: string) ->
    if (str == reverse(str)) "yes" else "no";

print("is 'racecar' a palindrome? ");
println(isPalindrome("racecar"));

print("is 'palindrome' a palindrome? ");
println(isPalindrome("palindrome"));

// recursive functions require explicit return types, normal functions don't
let factorial(n: int): int ->
    if (n == 0) 1
    else        n * factorial(n - 1);

// Turbina now supports lists as well
let answers: int[] = [
    factorial(0),
    factorial(5),
    factorial(10)
];
println(answers);
`;

const ADVANCED_SAMPLE_PROGRAM =
`// define a list like this
let numbers = [1, 2, 3, 4, 5, 6, 7];

// be creative with map, filter, reduce functions
// they work very similar to the JavaScript methods of the same name
let max(nums: int[]) -> reduce(
    nums,
    (x: int, y: int) -> if (x > y) x else y,
    0
);

let min(nums: int[]) -> reduce(
    nums,
    (x: int, y: int) -> if (x < y) x else y,
    9999999
);

// any type can be converted to string with the toString function
let results = [
    "numbers: " + toString(numbers),
    "max: " + toString(max(numbers)),
    "min: " + toString(min(numbers)),
    "asString: " + toString(
        reduce(numbers, (x: int, y: int) -> toString(x) + toString(y) + ";", "")
    ),
    "randNums: " + toString(map(numbers, (x: int) -> randInt(0, 10000)))
];
map(results, (result: string) -> println(result));
`;

export const SAMPLE_PROGRAMS = [
    BASIC_SAMPLE_PROGRAM,
    ADVANCED_SAMPLE_PROGRAM,
];
